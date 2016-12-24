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

func scaleRows(matrix autofunc.Result, scales autofunc.Result) autofunc.Result {
	return autofunc.Pool(scales, func(scales autofunc.Result) autofunc.Result {
		var res []autofunc.Result
		mags := autofunc.Split(len(scales.Output()), scales)
		for i, x := range autofunc.Split(len(mags), matrix) {
			res = append(res, autofunc.ScaleFirst(x, mags[i]))
		}
		return autofunc.Concat(res...)
	})
}

func scaleRowsR(matrix autofunc.RResult, scales autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(scales, func(scales autofunc.RResult) autofunc.RResult {
		var res []autofunc.RResult
		mags := autofunc.SplitR(len(scales.Output()), scales)
		for i, x := range autofunc.SplitR(len(mags), matrix) {
			res = append(res, autofunc.ScaleFirstR(x, mags[i]))
		}
		return autofunc.ConcatR(res...)
	})
}
