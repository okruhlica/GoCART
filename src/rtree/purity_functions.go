package rtree

type PurityFunction func(predictor string, t *rtree) (bestSplitIndex int, purityAtSplit float64)

// ====== Gini impurity measure

// Given the predictor and a tree node, this function returns the best split of observations according to Gini impurity measure.
// Output: a tuple containing the best split index and the combined gini impurity measure of the split (sum of impurities of both regions of the split)
// Side effects: observations in the node are reoredered in a sorted fashion according to values of provided predictor.
func GiniPurity(predictor string, t *rtree) (bestSplitIndex int, purityAtSplit float64) {
	t.SortByPredictor(predictor)
	goods := *calculateCummulativeGoodSlice(&t.Observations) // goods[i] <=> count of observations with target=1 in t.Observations[:i]
	sumGood := goods[len(goods)-1]
	countAll := len(goods)

	purityAtSplit, bestSplitIndex = NO_FLOAT, NO_INDEX
	for splitIndex, goodCount := range goods {
		countL, goodL := float64(splitIndex), float64(goodCount)
		countR, goodR := float64(countAll-splitIndex-1), float64(sumGood-goodCount)

		if int(countL) < t.GrowOptions.minSplitSize || int(countR) < t.GrowOptions.minSplitSize {
			continue
		}

		giniL := 1 - (goodL/countL)*(goodL/countL) - ((countL-goodL)/countL)*((countL-goodL)/countL)
		giniR := 1 - (goodR/countR)*(goodR/countR) - ((countR-goodR)/countR)*((countR-goodR)/countR)
		gini := (countL*giniL + countR*giniR)

		if purityAtSplit == NO_FLOAT || purityAtSplit > gini {
			purityAtSplit = gini
			bestSplitIndex = splitIndex
		}
	}
	purityAtSplit /= float64(countAll)
	return
}

// Returns the cummulative count of observations with target = 1 up to the given index (result is an array of ints).
func calculateCummulativeGoodSlice(observations *[]*Observation) *[]int {
	good := make([]int, len(*observations)+1)
	good[0] = 0
	for i, obs := range *observations {
		good[i+1] = good[i]
		if (*obs)[TARGET_KEY].Float == 1.0 {
			good[i+1]++
		}
	}
	return &good
}

// ======
