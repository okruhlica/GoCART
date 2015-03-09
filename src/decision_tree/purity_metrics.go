package decision_tree

// Blueprint interface for the purity measure calculation.
type AbstractPurityMetric interface {
	SplitPurity(predictor string, targetAttribute string, t *DecisionTree) (bestSplitIndex *int, purityAtSplit *float64, err error)
	SlicePurity(data []*Observation, targetAttribute string) (slicePurity float64, err error)
}

// --------------------------------------------------------------------------------------------------

// Gini purity measure
type GiniPurity struct{}

// Helper function to calculate the gini impurity of a set of observations.
func (g GiniPurity) SlicePurity(data []*Observation, targetAttribute string) (slicePurity float64, err error) {
	count, count1 := float64(len(data)), 0.0
	if count == 0.0 {
		return 0.0, nil
	}

	firstVal := (*data[0])[targetAttribute]
	for _, obs := range data {
		if isEq, err := _eq(firstVal, (*obs)[targetAttribute]); err != nil {
			return 1.0, err
		} else if isEq {
			count1 += 1.0
		}
	}

	p1 := count1 / count
	return 1 - p1*p1 - (1-p1)*(1-p1), nil
}

// Given the predictor and a tree node, this function returns the best split of observations according to Gini impurity measure.
// Output: a tuple containing the best split index and the combined gini impurity measure of the split (sum of impurities of both regions of the split)
// Side effects: observations in the node are reoredered in a sorted fashion according to values of provided predictor.
func (g GiniPurity) SplitPurity(predictor string, targetAttribute string, t *DecisionTree) (ptrBestSplitIndex *int, ptrPurityAtSplit *float64, err error) {
	t.sortByPredictor(predictor)
	goods, error := g.calculateCummulativeGoodSlice(t.Observations, targetAttribute) // goods[i] <=> count of observations with target=1 in t.Observations[:i]

	if error != nil {
		return
	}

	sumGood := (*goods)[len(*goods)-1]
	countAll := len(*goods)

	for splitIndex, goodCount := range *goods {
		if splitIndex < len(t.Observations)-1 {
			thisVal, nextVal := (*t.Observations[splitIndex])[predictor], (*t.Observations[splitIndex+1])[predictor]
			if isEq, _ := _eq(thisVal, nextVal); isEq { // only care about ends of "runs"
				continue
			}

		}

		countL, goodL := float64(splitIndex), float64(goodCount)
		countR, goodR := float64(countAll-splitIndex-1), float64(sumGood-goodCount)

		if int(countL) < t.Options.MinSplitSize || int(countR) < t.Options.MinSplitSize {
			continue
		}

		giniL := 1 - (goodL/countL)*(goodL/countL) - ((countL-goodL)/countL)*((countL-goodL)/countL)
		giniR := 1 - (goodR/countR)*(goodR/countR) - ((countR-goodR)/countR)*((countR-goodR)/countR)
		gini := (countL*giniL + countR*giniR)

		if ptrPurityAtSplit == nil || *ptrPurityAtSplit > gini {
			ptrPurityAtSplit = &gini
			i := splitIndex
			ptrBestSplitIndex = &i
		}
	}

	if ptrPurityAtSplit != nil {
		x := (*ptrPurityAtSplit / float64(countAll))
		ptrPurityAtSplit = &x
	}
	return
}

// Returns the cummulative count of observations with target = 1 up to the given index (result is an array of ints).
func (g GiniPurity) calculateCummulativeGoodSlice(observations []*Observation, target string) (*[]int, error) {
	if len(observations) == 0 {
		return &[]int{}, nil
	}

	firstVal := (*observations[0])[target]
	goods := make([]int, len(observations)+1)
	goods[0] = 0

	for i, obs := range observations {
		goods[i+1] = goods[i]
		if isEq, err := _eq(firstVal, (*obs)[target]); err != nil {
			return nil, err
		} else if isEq {
			goods[i+1]++
		}
	}
	return &goods, nil
}
