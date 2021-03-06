package decision_tree

type ByPredictorValueFloat struct {
	Predictor    string
	Observations *([]*Observation)
}

func (s ByPredictorValueFloat) Len() int {
	return len(*s.Observations)
}

func (s ByPredictorValueFloat) Swap(i, j int) {
	(*s.Observations)[i], (*s.Observations)[j] = (*s.Observations)[j], (*s.Observations)[i]
}

func (s ByPredictorValueFloat) Less(i, j int) bool {
	obsi, obsj := (*(*s.Observations)[i]), (*(*s.Observations)[j])
	isLess, _ := _lt(obsi[s.Predictor], obsj[s.Predictor])
	return isLess
}
