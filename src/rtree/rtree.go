package rtree

import "sort"
import "math"
import "strings"

const NO_PREDICTOR string = "__noPredictor__"
const NO_FLOAT = 9999999999.9937
const NO_INDEX = -1
const NO_CLASSIFICATION = -1

// Represents a classification tree (tree node)
type rtree struct {
	Left, Right, Up *rtree         // Pointers to parent and children nodes
	Observations    []*Observation // A slice of observations relevant to this node

	SplitPredictor *string // The predictor (attribute/feature) to split on in this node. Value in leaves is NO_PREDICTOR
	SplitValue     *Value  // The value of the split predictor to split on; smaller valued obserations continue to the left subtree, larger to the right subtree
	Impurity       float64 // Gini measure of impurity of the node (less is better)
	Classification int     // For leaf nodes denotes the predicted class; value is NO_CLASSIFICATION in internal nodes
	Depth          int     // Distance from tree root

	GrowOptions *GrowOptions // Settings governing the tree expansion and related stop-conditions.
}

// Returns true iff this node has no children
func (t *rtree) IsLeaf() bool {
	return t.Left == nil && t.Right == nil
}

// Initializes the provided node and sets the pointers so that it is the left child of the current node.
func (t *rtree) SetLeft(child *rtree) {
	child.InitNode(t.GrowOptions, t.Depth+1)
	child.Up = t
	t.Left = child
}

// Initializes the provided node and sets the pointers so that it is the right child of the current node.
func (t *rtree) SetRight(child *rtree) {
	child.InitNode(t.GrowOptions, t.Depth+1)
	child.Up = t
	t.Right = child
}

// Returns true iff the given attribute name is a valid for splitting upon.
// Only attributes that do not start with two underscores can be used as predictors.
func IsPredictor(predictor string) bool {
	return predictor != TARGET_KEY && !strings.HasPrefix(predictor, "__")
}

// Finds the best possible splitting paramters for the given node by testing all eligible splits on all eligible predictors.
// Returns: <predictor to split upon>  <index to split upon> <gini impurity of the split>
// Side-effects: Re-orders the observations within the node.
func (t *rtree) FindBestSplit() (bestPredictor string, bestIndex int, bestGini float64) {
	bestPredictor, bestIndex, bestGini = NO_PREDICTOR, NO_INDEX, NO_FLOAT
	for predictor := range *(t.Observations[0]) {
		if IsPredictor(predictor) {
			thisIdx, thisGini := t.BestSplitWithPredictor(predictor)
			if bestPredictor == NO_PREDICTOR || (thisGini < bestGini && thisGini != NO_FLOAT) {
				bestGini, bestIndex, bestPredictor = thisGini, thisIdx, predictor
			}
		}
	}
	return
}

// Sorts the observations in this node according to the values of the given predictor.
func (t *rtree) SortByPredictor(predictor string) {
	sortFunc := ByPredictorValueFloat{predictor, &t.Observations}
	sort.Sort(sortFunc)
}

func (t *rtree) BestSplitWithPredictor(predictor string) (splitAtIndex int, purityAtSplit float64) {
	return t.GrowOptions.purityFunction(predictor, t)
}

// Given the desired split position and an indicator which target class should be assigned to
// the left part of the slice, the function computes the classification error.
func (t *rtree) GiniOfSplit(splitIdx int) float64 {
	return gini(t.Observations[0:splitIdx]) + gini(t.Observations[splitIdx:])
}

// Helper function to calculate the gini impurity of a set of observations.
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

// Initializes the node so that it can be safely used within the tree.
func (t *rtree) InitNode(settings *GrowOptions, depth int) {
	t.GrowOptions = settings
	t.Impurity = gini(t.Observations)
	t.Classification = t.GetMajorityVote()
	t.Depth = depth
}

// Initializes the root node. Note: always call this function before first expanding on a node.
func (t *rtree) InitRoot(growOptions *GrowOptions, observations []*Observation) *rtree {
	t.Observations = observations
	t.InitNode(growOptions, 0)
	return t
}

// Expands the given node (if possible and allowed by the provided grow options setting) by finding the best split and performing it.
// If auto == true, this process is recursive, growing a full tree (one where calling Expand(*) on any of the nodes produces no further change).
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

	//	fmt.Printf("Splitting node [0..%d] at index %d around %s\n", len(t.Observations)-1, bestIndex, bestPredictor)
	//	fmt.Printf(serializeObservations(t.Observations) + "\n")
	t.SortByPredictor(bestPredictor)
	t.Split(bestIndex)

	t.SplitPredictor = &bestPredictor
	t.SplitValue = (*t.Observations[bestIndex])[bestPredictor]
	t.Classification = NO_CLASSIFICATION

	if auto {
		if t.Left != nil {
			t.Left.Expand(true)
		}
		if t.Right != nil {
			t.Right.Expand(true)
		}
	}
}

// Returns 0 iff at least half of the observations in this node have target value 0; 1 otherwise.
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

// Returns the classification for a new observation o.
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

// Returns a tuple describing the split rule for this node.
// Format for leaf nodes: <NO_PREDICTOR>,<NO_FLOAT>,<classification at node>
// Format for internal nodes: <split predictor>,<split value>,<NO_CLASSIFICATION>
func (t *rtree) GetRule() (predictor string, splitValue float64, classification int) {
	if t.IsLeaf() {
		return NO_PREDICTOR, NO_FLOAT, t.Classification
	} else {
		return *t.SplitPredictor, t.SplitValue.Float, NO_CLASSIFICATION
	}
}
