package weightnorm

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
)

func init() {
	var d denseCreator
	serializer.RegisterTypedDeserializer(d.SerializerType(), deserializeDenseCreator)
}

// NewDenseLayer creates a weight-normalized
// fully-connected layer based on a neuralnet.DenseLayer.
//
// The resulting *Norm is initialized to immitate d, only
// with its weight matrix decomposed into separate
// magnitude and direction vectors.
func NewDenseLayer(d *neuralnet.DenseLayer) *Norm {
	res := &Norm{
		Weights: []*autofunc.Variable{
			&autofunc.Variable{
				Vector: make(linalg.Vector, len(d.Weights.Data.Vector)),
			},
		},
		Mags: []*autofunc.Variable{
			&autofunc.Variable{
				Vector: make(linalg.Vector, d.OutputCount),
			},
		},
		Creator: &denseCreator{
			Biases:   d.Biases.Var,
			InCount:  d.InputCount,
			OutCount: d.OutputCount,
		},
	}
	copy(res.Weights[0].Vector, d.Weights.Data.Vector)
	cols := d.InputCount
	for row := 0; row < d.OutputCount; row++ {
		rowVec := res.Weights[0].Vector[row*cols : (row+1)*cols]
		res.Mags[0].Vector[row] = rowVec.Mag()
	}
	return res
}

type denseCreator struct {
	Biases   *autofunc.Variable
	InCount  int
	OutCount int
}

func deserializeDenseCreator(d []byte) (*denseCreator, error) {
	var res denseCreator
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (d *denseCreator) Create(p []*autofunc.Variable) neuralnet.Network {
	if len(p) != 1 {
		panic("expected exactly one parameter")
	}
	return neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  d.InCount,
			OutputCount: d.OutCount,
			Biases:      &autofunc.LinAdd{Var: d.Biases},
			Weights: &autofunc.LinTran{
				Data: p[0],
				Rows: d.OutCount,
				Cols: d.InCount,
			},
		},
	}
}

func (d *denseCreator) SerializerType() string {
	return "github.com/unixpickle/weightnorm.denseCreator"
}

func (d *denseCreator) Serialize() ([]byte, error) {
	return json.Marshal(d)
}
