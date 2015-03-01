package rtree

type SplitPurityFunction func(predictor string, t *Rtree) (bestSplitIndex int, purityAtSplit float64)
type PurityFunction func(observations []*Observation) float64

// ====== Gini impurity measure

// Helper function to calculate the gini impurity of a set of observations.
func GiniPurity(data []*Observation) float64 {
	count, count1 := len(data), 0
	if count == 0 {
		return 0.0
	}

	for _, obs := range data {
		if (*obs)[TARGET_KEY].Float == 0.0 {
			count1++
		}
	}

	p1 := float64(count1) / float64(count)
	return 1 - p1*p1 - (1-p1)*(1-p1)
}

// Given the predictor and a tree node, this function returns the best split of observations according to Gini impurity measure.
// Output: a tuple containing the best split index and the combined gini impurity measure of the split (sum of impurities of both regions of the split)
// Side effects: observations in the node are reoredered in a sorted fashion according to values of provided predictor.
func GiniSplitPurity(predictor string, t *Rtree) (bestSplitIndex int, purityAtSplit float64) {
	t.sortByPredictor(predictor)
	goods := *calculateCummulativeGoodSlice(&t.Observations) // goods[i] <=> count of observations with target=1 in t.Observations[:i]
	sumGood := goods[len(goods)-1]
	countAll := len(goods)

	purityAtSplit, bestSplitIndex = NO_FLOAT, NO_INDEX
	for splitIndex, goodCount := range goods {

		if splitIndex < len(t.Observations)-1 {
			thisVal, nextVal := (*t.Observations[splitIndex])[predictor].Float, (*t.Observations[splitIndex+1])[predictor].Float
			if thisVal == nextVal { // only care about ends of "runs"
				continue
			}
		}

		countL, goodL := float64(splitIndex), float64(goodCount)
		countR, goodR := float64(countAll-splitIndex-1), float64(sumGood-goodCount)

		if int(countL) < t.GrowOptions.MinSplitSize || int(countR) < t.GrowOptions.MinSplitSize {
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
