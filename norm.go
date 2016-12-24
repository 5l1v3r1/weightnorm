package weightnorm

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var n Norm
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNorm)
}

// NetCreator creates networks for a Norm instance.
type NetCreator interface {
	serializer.Serializer
	Create(params []*autofunc.Variable) neuralnet.Network
}

// Norm applies weight normalization by normalizing a
// certain set of parameters, feeding them into a function
// which produces a neuralnet.Network, and then evaluating
// the resulting network.
type Norm struct {
	// Weights is the list of unnormalized weight arrays.
	Weights []*autofunc.Variable

	// Mags stores the magnitude scalers for each of the
	// weight arrays.
	// An entry Mags[i] contains one component per weight
	// vector in Weights[i].
	// Thus, the length of Mags[i] indicates how many vectors
	// are packed in Weights[i].
	Mags []*autofunc.Variable

	// Creator creates a network from a set of normalized
	// parameters.
	Creator NetCreator
}

// DeserializeNorm deserializes a Norm.
func DeserializeNorm(d []byte) (*Norm, error) {
	var weights serializer.Bytes
	var mags serializer.Bytes
	var res Norm

	if err := serializer.DeserializeAny(d, &weights, &mags, &res.Creator); err != nil {
		return nil, err
	}

	if err := json.Unmarshal(weights, &res.Weights); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(mags, &res.Mags); err != nil {
		return nil, err
	}

	return &res, nil
}

// Apply generates a network and applies it to the input.
func (n *Norm) Apply(in autofunc.Result) autofunc.Result {
	res := n.newNormResult()
	net := n.Creator.Create(res.NormPool)
	res.Result = net.Apply(in)
	return res
}

// ApplyR is like Apply, but for RResults.
func (n *Norm) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	res, newRV := n.newNormRResult(rv)
	net := n.Creator.Create(res.NormPool)
	res.Result = net.ApplyR(newRV, in)
	return res
}

// Batch applies the network in batch.
func (n *Norm) Batch(in autofunc.Result, m int) autofunc.Result {
	res := n.newNormResult()
	net := n.Creator.Create(res.NormPool)
	res.Result = net.BatchLearner().Batch(in, m)
	return res
}

// BatchR applies the network in batch.
func (n *Norm) BatchR(rv autofunc.RVector, in autofunc.RResult, m int) autofunc.RResult {
	res, newRV := n.newNormRResult(rv)
	net := n.Creator.Create(res.NormPool)
	res.Result = net.BatchLearner().BatchR(newRV, in, m)
	return res
}

// Parameters generates a list of learning parameters by
// including not only the weights and magnitudes, but also
// the parameters of a generated network (not including
// the temporary normalized variables).
//
// The parameters are ordered as follows: n.Weights,
// n.Mags, network.Parameters().
func (n *Norm) Parameters() []*autofunc.Variable {
	res := make([]*autofunc.Variable, len(n.Weights)+len(n.Mags))
	copy(res, n.Weights)
	copy(res[len(n.Weights):], n.Mags)

	normRes, rv := n.newNormRResult(autofunc.RVector{})
	net := n.Creator.Create(normRes.NormPool)
	params := net.Parameters()
	for _, param := range params {
		if _, ok := rv[param]; !ok {
			res = append(res, param)
		}
	}

	return res
}

// SerializerType returns the unique ID used to serialize
// a Norm with the serializer package.
func (n *Norm) SerializerType() string {
	return "github.com/unixpickle/weightnorm.Norm"
}

// Serialize serializes the Norm.
func (n *Norm) Serialize() ([]byte, error) {
	weightsData, err := json.Marshal(n.Weights)
	if err != nil {
		return nil, err
	}
	magsData, err := json.Marshal(n.Mags)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeAny(
		serializer.Bytes(weightsData),
		serializer.Bytes(magsData),
		n.Creator,
	)
}

func (n *Norm) normalize() []autofunc.Result {
	var res []autofunc.Result
	for i, weights := range n.Weights {
		mags := n.Mags[i]
		norms := rowNorms(weights, len(mags.Vector))
		scales := autofunc.Mul(mags, autofunc.Inverse(norms))
		res = append(res, scaleRows(weights, scales))
	}
	return res
}

func (n *Norm) normalizeR(rv autofunc.RVector) []autofunc.RResult {
	var res []autofunc.RResult
	for i, weightsVar := range n.Weights {
		weights := autofunc.NewRVariable(weightsVar, rv)
		mags := autofunc.NewRVariable(n.Mags[i], rv)
		norms := rowNormsR(weights, len(mags.Output()))
		scales := autofunc.MulR(mags, autofunc.InverseR(norms))
		res = append(res, scaleRowsR(weights, scales))
	}
	return res
}

func (n *Norm) newNormResult() *normResult {
	res := &normResult{
		Norm:     n,
		NormRes:  n.normalize(),
		NormPool: make([]*autofunc.Variable, len(n.Weights)),
	}
	for i, x := range res.NormRes {
		res.NormPool[i] = &autofunc.Variable{Vector: x.Output()}
	}
	return res
}

func (n *Norm) newNormRResult(rv autofunc.RVector) (*normRResult, autofunc.RVector) {
	newRV := autofunc.RVector{}
	for k, v := range rv {
		newRV[k] = v
	}
	res := &normRResult{
		Norm:     n,
		NormRes:  n.normalizeR(rv),
		NormPool: make([]*autofunc.Variable, len(n.Weights)),
	}
	for i, x := range res.NormRes {
		res.NormPool[i] = &autofunc.Variable{Vector: x.Output()}
		newRV[res.NormPool[i]] = x.ROutput()
	}
	return res, newRV
}

type normResult struct {
	Norm     *Norm
	NormRes  []autofunc.Result
	NormPool []*autofunc.Variable
	Result   autofunc.Result
}

func (n *normResult) Output() linalg.Vector {
	return n.Result.Output()
}

func (n *normResult) Constant(g autofunc.Gradient) bool {
	if !n.Result.Constant(g) {
		return false
	}
	if n.rawParamsConstant(g) {
		return true
	}
	gPool := autofunc.NewGradient(n.NormPool)
	return n.Result.Constant(gPool)
}

func (n *normResult) PropagateGradient(u linalg.Vector, g autofunc.Gradient) {
	if n.Constant(g) {
		return
	}
	for _, p := range n.NormPool {
		g[p] = make(linalg.Vector, len(p.Vector))
	}
	n.Result.PropagateGradient(u, g)
	poolUpstream := make([]linalg.Vector, len(n.NormPool))
	for i, p := range n.NormPool {
		poolUpstream[i] = g[p]
		delete(g, p)
	}
	for i, up := range poolUpstream {
		n.NormRes[i].PropagateGradient(up, g)
	}
}

func (n *normResult) rawParamsConstant(g autofunc.Gradient) bool {
	for _, list := range [][]*autofunc.Variable{n.Norm.Weights, n.Norm.Mags} {
		for _, x := range list {
			if _, ok := g[x]; ok {
				return false
			}
		}
	}
	return true
}

type normRResult struct {
	Norm     *Norm
	NormRes  []autofunc.RResult
	NormPool []*autofunc.Variable
	Result   autofunc.RResult
}

func (n *normRResult) Output() linalg.Vector {
	return n.Result.Output()
}

func (n *normRResult) ROutput() linalg.Vector {
	return n.Result.ROutput()
}

func (n *normRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	if !n.Result.Constant(rg, g) {
		return false
	}
	if n.rawParamsConstant(rg, g) {
		return true
	}
	rgPool := autofunc.NewRGradient(n.NormPool)
	return n.Result.Constant(rgPool, nil)
}

func (n *normRResult) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if n.Constant(rg, g) {
		return
	}
	if g == nil {
		g = autofunc.Gradient{}
	}
	for _, p := range n.NormPool {
		g[p] = make(linalg.Vector, len(p.Vector))
		rg[p] = make(linalg.Vector, len(p.Vector))
	}
	n.Result.PropagateRGradient(u, uR, rg, g)
	poolUpstream := make([]linalg.Vector, len(n.NormPool))
	poolUpstreamR := make([]linalg.Vector, len(n.NormPool))
	for i, p := range n.NormPool {
		poolUpstream[i] = g[p]
		poolUpstreamR[i] = rg[p]
		delete(g, p)
		delete(rg, p)
	}
	for i, up := range poolUpstream {
		n.NormRes[i].PropagateRGradient(up, poolUpstreamR[i], rg, g)
	}
}

func (n *normRResult) rawParamsConstant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	for _, list := range [][]*autofunc.Variable{n.Norm.Weights, n.Norm.Mags} {
		for _, x := range list {
			if g != nil {
				if _, ok := g[x]; ok {
					return false
				}
			}
			if _, ok := rg[x]; ok {
				return false
			}
		}
	}
	return true
}
