package weightnorm

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

func BenchmarkRowNorms(b *testing.B) {
	mat := &autofunc.Variable{
		Vector: make(linalg.Vector, 300*500),
	}
	for i := range mat.Vector {
		mat.Vector[i] = rand.NormFloat64()
	}
	upstream := make(linalg.Vector, 500)
	for i := range upstream {
		upstream[i] = rand.NormFloat64()
	}
	grad := autofunc.NewGradient([]*autofunc.Variable{mat})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		u := append(linalg.Vector{}, upstream...)
		rowNorms(mat, len(upstream)).PropagateGradient(u, grad)
	}
}

func BenchmarkScaleRows(b *testing.B) {
	mat := &autofunc.Variable{
		Vector: make(linalg.Vector, 300*500),
	}
	upstream := make(linalg.Vector, 300*500)
	for i := range mat.Vector {
		mat.Vector[i] = rand.NormFloat64()
		upstream[i] = rand.NormFloat64()
	}

	scale := &autofunc.Variable{
		Vector: make(linalg.Vector, 500),
	}
	for i := range scale.Vector {
		scale.Vector[i] = rand.NormFloat64()
	}

	grad := autofunc.NewGradient([]*autofunc.Variable{mat, scale})
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		u := append(linalg.Vector{}, upstream...)
		scaleRows(mat, scale).PropagateGradient(u, grad)
	}
}
