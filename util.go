package weightnorm

import "github.com/unixpickle/autofunc"

func rowNorms(matrix autofunc.Result, numRows int) autofunc.Result {
	var res []autofunc.Result
	for _, x := range autofunc.Split(numRows, matrix) {
		norm := autofunc.SquaredNorm{}.Apply(x)
		res = append(res, autofunc.Pow(norm, 0.5))
	}
	return autofunc.Concat(res...)
}

func rowNormsR(matrix autofunc.RResult, numRows int) autofunc.RResult {
	var res []autofunc.RResult
	for _, x := range autofunc.SplitR(numRows, matrix) {
		norm := autofunc.SquaredNorm{}.ApplyR(nil, x)
		res = append(res, autofunc.PowR(norm, 0.5))
	}
	return autofunc.ConcatR(res...)
}
