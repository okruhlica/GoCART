package rtree

import "sort"
import "math"

const NO_PREDICTOR string = "__noPredictor__"
const NO_FLOAT = 9999999999.9937
const NO_INDEX = -1
const NO_CLASSIFICATION = -1

// Represents a classification tree (tree node)
type Rtree struct {
	Left, Right, Up *Rtree         // Pointers to parent and children nodes
	Observations    []*Observation // A slice of observations relevant to this node

	SplitPredictor *string // The predictor (attribute/feature) to split on in this node. Value in leaves is NO_PREDICTOR
	SplitValue     *Value  // The value of the split predictor to split on; smaller valued obserations continue to the left subtree, larger to the right subtree
	Impurity       float64 // Gini measure of impurity of the node (less is better)
	Classification int     // For leaf nodes denotes the predicted class; value is NO_CLASSIFICATION in internal nodes
	Depth          int     // Distance from tree root

	GrowOptions *GrowOptions // Settings governing the tree expansion and related stop-conditions.
}

// Initializes the provided node and sets the pointers so that it is the left child of the current node.
func (t *Rtree) setLeft(child *Rtree) {
	child.initNode(t.GrowOptions, t.Depth+1)
	child.Up = t
	t.Left = child
}

// Initializes the provided node and sets the pointers so that it is the right child of the current node.
func (t *Rtree) setRight(child *Rtree) {
	child.initNode(t.GrowOptions, t.Depth+1)
	child.Up = t
	t.Right = child
}

func getSortedPredictorsFromObservation(obs *Observation) *[]string {
	predictors := make([]string, 0, len(*obs))
	for pred, _ := range *obs {
		predictors = append(predictors, pred)
	}

	sort.Strings(predictors)
	return &predictors
}

// Finds the best possible splitting paramters for the given node by testing all eligible splits on all eligible predictors.
// Returns: <predictor to split upon>  <index to split upon> <gini impurity of the split>
// Side-effects: Re-orders the observations within the node.
func (t *Rtree) FindBestSplit() (bestPredictor string, bestIndex int, bestGini float64) {
	bestPredictor, bestIndex, bestGini = NO_PREDICTOR, NO_INDEX, NO_FLOAT

	predictors := getSortedPredictorsFromObservation(t.Observations[0])

	for _, predictor := range *predictors {
		if isPredictor(predictor) {
			thisIdx, thisGini := t.bestSplitWithPredictor(predictor)
			if bestPredictor == NO_PREDICTOR || (thisGini < bestGini && thisGini != NO_FLOAT) {
				bestGini, bestIndex, bestPredictor = thisGini, thisIdx, predictor
			}
		}
	}
	return
}

// Sorts the observations in this node according to the values of the given predictor.
func (t *Rtree) sortByPredictor(predictor string) {
	sortFunc := ByPredictorValueFloat{predictor, &t.Observations}
	sort.Sort(sortFunc)
}

func (t *Rtree) bestSplitWithPredictor(predictor string) (splitAtIndex int, purityAtSplit float64) {
	return t.GrowOptions.SplitStrategy.GetSplitPurity(predictor, t)
}

// Initializes the node so that it can be safely used within the tree.
func (t *Rtree) initNode(settings *GrowOptions, depth int) {
	t.GrowOptions = settings
	fn := settings.SplitStrategy.GetSlicePurity
	t.Impurity = fn(t.Observations[:])
	t.Classification = t.getMajorityVote()
	t.Depth = depth
}

// Initializes the root node. Note: always call this function before first expanding on a node.
func (t *Rtree) InitRoot(growOptions *GrowOptions, observations []*Observation) *Rtree {
	t.Observations = observations
	t.initNode(growOptions, 0)
	return t
}

// Expands the given node (if possible and allowed by the provided grow options setting) by finding the best split and performing it.
// If auto == true, this process is recursive, growing a full tree (one where calling Expand(*) on any of the nodes produces no further change).
func (t *Rtree) Expand(auto bool) {
	// check for grow-stop conditions
	if len(t.Observations) < t.GrowOptions.MinSplitSize ||
		t.Impurity < t.GrowOptions.MaxSplitGini ||
		t.Depth >= t.GrowOptions.MaxDepth {
		t.Classification = t.getMajorityVote()
		return
	}

	// try to find a predictor
	bestPredictor, bestIndex, _ := t.FindBestSplit()
	if bestPredictor == NO_PREDICTOR || bestIndex == NO_INDEX || bestIndex == 0 || bestIndex == len(t.Observations) {
		t.Classification = t.getMajorityVote()
		return
	}

	t.sortByPredictor(bestPredictor)
	t.split(bestIndex)

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
func (t *Rtree) getMajorityVote() int {
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
func (t *Rtree) split(splitIdx int) {
	if splitIdx < 0 || splitIdx >= len(t.Observations) {
		return
	}

	l := new(Rtree)
	r := new(Rtree)

	l.Observations = t.Observations[:splitIdx]
	r.Observations = t.Observations[splitIdx:]

	t.setLeft(l)
	t.setRight(r)
}

// Returns the classification for a new observation o.
func (t *Rtree) Classify(o *Observation) int {
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
func (t *Rtree) GetRule() (predictor string, splitValue float64, classification int) {
	if t.IsLeaf() {
		return NO_PREDICTOR, NO_FLOAT, t.Classification
	} else {
		return *t.SplitPredictor, t.SplitValue.Float, NO_CLASSIFICATION
	}
}
