package rtree

import "sort"
import "math"
import "strings"
import "fmt"

const NO_PREDICTOR string = "__noPredictor__"
const NO_FLOAT = 9999999999.9937
const NO_GINI_VALUE float64 = NO_FLOAT
const NO_INDEX = -1
const NO_CLASSIFICATION = -1

type rtree struct {
	Left, Right, Up *rtree
	Observations    []*Observation

	SplitPredictor *string
	SplitValue     *Value
	Impurity       float64
	Classification int
	Depth          int

	GrowOptions *GrowOptions
}

func (t *rtree) IsLeaf() bool {
	return t.Left == nil && t.Right == nil
}

func (t *rtree) SetLeft(child *rtree) {
	child.InitNode(t.GrowOptions, t.Depth+1)
	child.Up = t
	t.Left = child
}

func (t *rtree) SetRight(child *rtree) {
	child.InitNode(t.GrowOptions, t.Depth+1)
	child.Up = t
	t.Right = child
}

func (t *rtree) GetLeaves() []*rtree {
	if t.IsLeaf() {
		return []*rtree{t}
	}

	leaves := []*rtree{}
	if t.Left != nil {
		leaves = t.Left.GetLeaves()
	}

	if t.Right != nil {
		return append(leaves, t.Right.GetLeaves()...)
	}
	return leaves
}

func isEligiblePredictor(predictor string) bool {
	return predictor != TARGET_KEY && !strings.HasPrefix(predictor, "__")
}

func (t *rtree) FindBestSplit() (string, int, float64) {
	bestPredictor, bestIndex, bestGini := NO_PREDICTOR, NO_INDEX, NO_GINI_VALUE
	for predictor := range *(t.Observations[0]) {
		if isEligiblePredictor(predictor) {
			thisIdx, thisGini := t.BestSplitWithPredictor(predictor)
			if bestPredictor == NO_PREDICTOR || (thisGini < bestGini && thisGini != NO_GINI_VALUE) {
				bestGini, bestIndex, bestPredictor = thisGini, thisIdx, predictor
			}
		}
	}
	return bestPredictor, bestIndex, bestGini
}

// Sorts the observations in this node according to the values of the given predictor.
func (t *rtree) SortByPredictor(predictor string) {
	sortFunc := ByPredictorValueFloat{predictor, &t.Observations}
	sort.Sort(sortFunc)
}

// Calculates the cummulative count of observations with target = 1 up to the given index and returns an array of ints.
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

// Given the predictor and a tree node, this function returns the best split of observations according to Gini impurity measure.
// Output: a tuple containing the best split index and the combined gini impurity measure of the split (sum of impurities of both regions of the split)
// Side effects: observations in the node are reoredered in a sorted fashion according to values of provided predictor.
func (t *rtree) BestSplitWithPredictor(predictor string) (int, float64) {

	t.SortByPredictor(predictor)
	goods := *calculateCummulativeGoodSlice(&t.Observations) // goods[i] <=> count of observations with target=1 in t.Observations[:i]
	sumGood := goods[len(goods)-1]
	countAll := len(goods)

	bestGini, bestIndex := NO_GINI_VALUE, NO_INDEX
	for splitIndex, goodCount := range goods {
		countL, goodL := float64(splitIndex), float64(goodCount)
		countR, goodR := float64(countAll-splitIndex-1), float64(sumGood-goodCount)

		if int(countL) < t.GrowOptions.minSplitSize || int(countR) < t.GrowOptions.minSplitSize {
			continue
		}

		giniL := 1 - (goodL/countL)*(goodL/countL) - ((countL-goodL)/countL)*((countL-goodL)/countL)
		giniR := 1 - (goodR/countR)*(goodR/countR) - ((countR-goodR)/countR)*((countR-goodR)/countR)
		gini := 1.0 / ((countL / giniL) + (countR / giniR))
		if bestGini == NO_GINI_VALUE || bestGini > gini {
			bestGini = gini
			bestIndex = splitIndex
		}
	}

	return bestIndex, bestGini
}

// Given the desired split position and an indicator which target class should be assigned to
// the left part of the slice, the function computes the classification error.
func (t *rtree) GiniOfSplit(splitIdx int) float64 {
	return gini(t.Observations[0:splitIdx]) + gini(t.Observations[splitIdx:])
}

func gini(data []*Observation) float64 {
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

func (t *rtree) InitNode(settings *GrowOptions, depth int) {
	t.GrowOptions = settings
	t.Impurity = gini(t.Observations)
	t.Classification = t.GetMajorityVote()
	t.Depth = depth
}

func (t *rtree) Expand(auto bool) {
	// check for grow-stop conditions
	if len(t.Observations) < t.GrowOptions.minSplitSize ||
		t.Impurity < t.GrowOptions.maxSplitGini ||
		t.Depth >= t.GrowOptions.maxDepth {
		t.Classification = t.GetMajorityVote()
		return
	}

	// try to find a predictor
	bestPredictor, bestIndex, _ := t.FindBestSplit()
	if bestPredictor == NO_PREDICTOR || bestIndex == NO_INDEX || bestIndex == 0 || bestIndex == len(t.Observations) {
		t.Classification = t.GetMajorityVote()
		return
	}

	fmt.Printf("Splitting node [0..%d] at index %d around %s\n", len(t.Observations)-1, bestIndex, bestPredictor)
	//	fmt.Printf(serializeObservations(t.Observations) + "\n")
	t.SortByPredictor(bestPredictor)
	t.Split(bestIndex)

	t.SplitPredictor = &bestPredictor
	t.SplitValue = (*t.Observations[bestIndex])[bestPredictor]
	t.Classification = NO_CLASSIFICATION

	if auto {
		t.Left.Expand(true)
		t.Right.Expand(true)
	}
}

func (t *rtree) GetMajorityVote() int {
	count, count1 := len(t.Observations), 0
	for _, obs := range t.Observations {
		if math.Abs((*obs)[TARGET_KEY].Float-1.0) < 0.001 {
			count1++
		}
	}

	if count1 > count-count1 {
		return 1
	}
	return 0
}

// Splits the node into subtrees so that values [0:splitIdx-1] belong to the left subtree and [splitIdx:] to the right subtree.
func (t *rtree) Split(splitIdx int) {
	if splitIdx < 0 || splitIdx >= len(t.Observations) {
		return
	}

	l := new(rtree)
	r := new(rtree)

	l.Observations = t.Observations[:splitIdx]
	r.Observations = t.Observations[splitIdx:]

	t.SetLeft(l)
	t.SetRight(r)
}

func (t *rtree) Classify(o *Observation) int {
	feature, val, classification := t.GetRule()

	if classification != NO_CLASSIFICATION {
		return t.Classification
	}

	if (*(*o)[feature]).Float < val {
		return t.Left.Classify(o)
	}
	return t.Right.Classify(o)
}

func (t *rtree) GetRule() (string, float64, int) {
	if t.IsLeaf() {
		return NO_PREDICTOR, NO_FLOAT, t.Classification
	} else {
		return *t.SplitPredictor, t.SplitValue.Float, NO_CLASSIFICATION
	}
}
