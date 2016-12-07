package weightnorm

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

func TestDenseLayerOutput(t *testing.T) {
	dl := &neuralnet.DenseLayer{
		InputCount:  3,
		OutputCount: 5,
	}
	dl.Randomize()
	norm := NewDenseLayer(dl)

	in := &autofunc.Variable{Vector: make(linalg.Vector, dl.InputCount)}
	for i := range in.Vector {
		in.Vector[i] = rand.NormFloat64()
	}

	expected := dl.Apply(in)
	actual := norm.Apply(in)

	if expected.Output().Copy().Scale(-1).Add(actual.Output()).MaxAbs() > 1e-5 {
		t.Errorf("expected %v but got %v", expected.Output(), actual.Output())
	}
}

func TestDenseLayer(t *testing.T) {
	dl := &neuralnet.DenseLayer{
		InputCount:  3,
		OutputCount: 5,
	}
	dl.Randomize()
	norm := NewDenseLayer(dl)
	if len(norm.Parameters()) != 3 {
		t.Fatalf("unexpected parameter count: %d", len(norm.Parameters()))
	}

	in := &autofunc.Variable{Vector: make(linalg.Vector, dl.InputCount)}
	for i := range in.Vector {
		in.Vector[i] = rand.NormFloat64()
	}

	rv := autofunc.RVector{}
	for _, v := range norm.Parameters() {
		rv[v] = make(linalg.Vector, len(v.Vector))
		for i := range rv[v] {
			rv[v][i] = rand.NormFloat64()
		}
	}

	checker := functest.RFuncChecker{
		F:     norm,
		Vars:  norm.Parameters(),
		Input: in,
		RV:    rv,
	}
	checker.FullCheck(t)
}
