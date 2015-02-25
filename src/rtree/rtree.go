package rtree

import "sort"
import "math"

type rtree struct {
	Left, Right, Up *rtree
	Observations    []*Observation

	SplitPredictor *string
	SplitValue     *Value
	Impurity       float64
	Classification int
}

func (t *rtree) IsLeaf() bool {
	return t.Left == nil && t.Right == nil
}

func (t *rtree) SetLeft(child *rtree) {
	child.InitRoot()
	child.Up = t
	t.Left = child
}

func (t *rtree) SetRight(child *rtree) {
	child.InitRoot()
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

const NO_PREDICTOR string = "__noPredictor__"
const NO_GINI_VALUE float64 = 99999999999999.0
const NO_INDEX = -1
const NO_CLASSIFICATION = -1

func (t *rtree) FindBestSplit() (string, int, float64) {
	bestPredictor, bestIndex, bestGini := NO_PREDICTOR, NO_INDEX, NO_GINI_VALUE
	for predictor := range *(t.Observations[0]) {
		if predictor != TARGET_KEY {
			thisIdx, thisGini := t.FindPredictorSplit(predictor)
			if bestPredictor == NO_PREDICTOR || thisGini < bestGini {
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

func (t *rtree) FindPredictorSplit(predictor string) (int, float64) {
	t.SortByPredictor(predictor)
	bestGini, bestIndex := NO_GINI_VALUE, NO_INDEX
	for splitIdx := 0; splitIdx < len(t.Observations); splitIdx++ {
		gini := t.GiniOfSplit(splitIdx)
		if gini < bestGini || bestIndex == NO_INDEX {
			bestGini, bestIndex = gini, splitIdx
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

func (t *rtree) InitRoot() {
	t.Impurity = gini(t.Observations)
	t.Classification = t.GetMajorityVote()
}

func (t *rtree) Expand(auto bool) {
	if len(t.Observations) <= 1 {
		t.Classification = t.GetMajorityVote()
		return
	}

	bestPredictor, bestIndex, _ := t.FindBestSplit()
	if bestPredictor == NO_PREDICTOR || bestIndex == 0 {
		return
	}
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

	t.Left.Impurity = gini(l.Observations)
	t.Right.Impurity = gini(r.Observations)
}

func (t *rtree) Classify(o *Observation) int {
	if t.Classification != NO_CLASSIFICATION {
		return t.Classification
	}

	if (*(*o)[*t.SplitPredictor]).Float < t.SplitValue.Float {
		return t.Left.Classify(o)
	}
	return t.Right.Classify(o)
}
