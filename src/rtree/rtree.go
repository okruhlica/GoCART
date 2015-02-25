package rtree

import "sort"
import "fmt"
import "strings"
import "strconv"

type rtree struct {
	Left, Right, Up *rtree
	Observations    []*Observation

	SplitPredictor *string
	SplitValue     *Value
	Impurity       float64
}

func (t *rtree) IsLeaf() bool {
	return t.Left == nil && t.Right == nil
}

func (t *rtree) SetLeft(child *rtree) {
	child.Up = t
	t.Left = child
}

func (t *rtree) SetRight(child *rtree) {
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
}

func (t *rtree) Expand(auto bool) {
	if len(t.Observations) <= 1 {
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

	if auto {
		t.Left.Expand(true)
		t.Right.Expand(true)
	}
}

// Splits the node into subtrees so that values [0:splitIdx-1] belong to the left subtree and [splitIdx:] to the right subtree.
func (t *rtree) Split(splitIdx int) {
	if splitIdx < 0 || splitIdx >= len(t.Observations) {
		return
	}

	l := new(rtree)
	r := new(rtree)

	l.Observations = t.Observations[0:splitIdx]
	r.Observations = t.Observations[splitIdx:]

	t.SetLeft(l)
	t.SetRight(r)

	t.Left.Impurity = gini(l.Observations)
	t.Right.Impurity = gini(r.Observations)
}

func (t *rtree) PrintTree(depth int) string {
	if t != nil {
		splitPredictor, splitVal := NO_PREDICTOR, -1.0
		if t.SplitValue != nil && t.SplitPredictor != nil {
			splitVal = t.SplitValue.Float
			splitPredictor = *t.SplitPredictor
		}
		fmt.Printf("%s (rule: %s < %f, purity: %f)[%d observations:[%s]]\n", strings.Repeat("-", depth*3+1),
			splitPredictor,
			splitVal,
			t.Impurity,
			len(t.Observations),
			serializeObservations(t.Observations))
		return t.Left.PrintTree(depth+1) + t.Right.PrintTree(depth+1)
	}
	return ""
}

func serializeObservations(observations []*Observation) string {
	out := ""
	for _, obs := range observations {
		out += ",target=" + strconv.FormatFloat((*obs)[TARGET_KEY].Float, 'f', 6, 64)
	}
	return out
}
