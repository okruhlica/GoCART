package decision_tree

import (
	"errors"
	"sort"
)

const NO_PREDICTOR string = "__noPredictor__"
const NO_FLOAT = 9999999999.9937
const NO_INDEX = -1

// Represents a classification tree (tree node)
type DecisionTree struct {
	left, right  *DecisionTree  // Pointers to parent and children nodes
	Observations []*Observation // A slice of observations relevant to this node
	Options      *Options       // Settings governing the tree expansion and related stop-conditions.

	Depth int // Distance from tree root

	SplitPredictor *string // The predictor (attribute/feature) to split on in this node. Value in leaves is NO_PREDICTOR
	SplitValue     Value   // The value of the split predictor to split on; smaller valued obserations continue to the left subtree, larger to the right subtree
	Classification Value   // For leaf nodes denotes the predicted class; value is NO_CLASSIFICATION in internal nodes

	impurity  *float64 // Measure of the node (less is better)
	_sortedBy *string
}

// Initializes the provided node and sets the pointers so that it is the left child of the current node.
func (t *DecisionTree) setLeft(child *DecisionTree) {
	child.initNode(t.Options, t.Depth+1)
	t.left = child
}

// Initializes the provided node and sets the pointers so that it is the right child of the current node.
func (t *DecisionTree) setRight(child *DecisionTree) {
	child.initNode(t.Options, t.Depth+1)
	t.right = child
}

// Returns the left subtree (if existent).
func (t *DecisionTree) Left() *DecisionTree {
	return t.left
}

// Returns the right subtree (if existent).
func (t *DecisionTree) Right() *DecisionTree {
	return t.right
}

// Finds the best possible splitting parameters for the given node by testing all eligible splits on all eligible predictors.
// Returns: <predictor to split upon>  <index to split upon> <gini impurity of the split>
// Side-effects: Re-orders the observations within the node.
func (t *DecisionTree) FindBestSplit() (bestPredictor string, bestIndex *int, bestPurity *float64, err error) {
	bestPredictor, bestIndex, bestPurity = NO_PREDICTOR, nil, nil

	for _, predictor := range *t.Options.Predictors {
		index, purity, err1 := t.bestSplitWithPredictor(predictor)
		if err1 == nil {
			if (bestPurity == nil && purity != nil) || (purity != nil && *purity < *bestPurity) {
				bestPurity, bestIndex, bestPredictor = purity, index, predictor
			}
		} else {
			err = err1
			return
		}
	}
	return
}

// Sorts the observations in this node according to the values of the given predictor.
func (t *DecisionTree) sortByPredictor(predictor string) {
	if t._sortedBy == nil || *t._sortedBy != predictor {
		sortFunc := ByPredictorValueFloat{predictor, &t.Observations}
		sort.Sort(sortFunc)
		t._sortedBy = &predictor
	}
}

func (t *DecisionTree) bestSplitWithPredictor(predictor string) (splitAtIndex *int, purityAtSplit *float64, err error) {
	return t.Options.SplitStrategy.SplitPurity(predictor, t.Options.TargetAttribute, t)
}

// Initializes the node so that it can be safely used within the tree.
func (t *DecisionTree) initNode(settings *Options, depth int) {
	t.Options = settings
	t.Depth = depth
}

// Impurity returns a float64 value describing how (un)well this node splits the domain according to the target attribute.
// This method is a wrapper over the injected SplitStrategy's purity methods, only assuming that it returns smaller impurity values for
// more ordered sets, so this is the one and only guarantee this method can give.
// Moreover, this method is lazy in terms of impurity calculations, as it calculates the impurity of the node only once upon first invocation,
// and returns the memoized impurity value from all subsequent calls to this method.
func (t *DecisionTree) Impurity() (float64, error) {
	if t.impurity != nil {
		return *t.impurity, nil
	}
	fn := t.Options.SplitStrategy.SlicePurity
	val, err := fn(t.Observations[:], t.Options.TargetAttribute)
	t.impurity = &val
	return *t.impurity, err
}

// Initializes the root node. Note: always call this function before first expanding on a node.
func (t *DecisionTree) InitRoot(growOptions *Options, observations []*Observation) error {
	if len(observations) == 0 {
		return nil
	}

	if !t.testPredictorsPresent() {
		return errors.New("Not all observations contain values for all predictors.")
	}

	t.Observations = observations
	t.initNode(growOptions, 0)
	return nil
}

func (t *DecisionTree) testPredictorsPresent() bool {
	for _, obs := range t.Observations {
		for _, pred := range *t.Options.Predictors {
			if _, ok := (*obs)[pred]; !ok {
				return false
			}
		}
	}
	return true
}

func (t *DecisionTree) isGrowable() (bool, error) {
	if _, err := t.Impurity(); err != nil {
		return false, err
	}

	// check for grow-stop conditions
	N := len(t.Observations)
	options := t.Options

	tooDeep := t.Depth >= options.MaxDepth
	pureEnough := *t.impurity < options.MaxSplitImpurity
	tooSpecific := N < options.MinSplitSize
	return !tooDeep && !pureEnough && !tooSpecific, nil
}

// expands the given node (if possible and allowed by the provided grow options setting) by finding the best split and performing it.
// If auto == true, this process is recursive, growing a full tree (one where calling Expand(*) on any of the nodes produces no further change).
func (t *DecisionTree) Expand(auto bool) (err error) {

	// Find out if we are allowed to expand the tree one more level
	if canGrow, err := t.isGrowable(); err != nil {
		return err
	} else if !auto || !canGrow {
		t.Classification, err = t.getMajorityVote()
		return err
	}

	// try to find a predictor and either classify if no further split is available, or split
	if bestPredictor, bestIndex, _, err := t.FindBestSplit(); err != nil {
		return err
	} else if bestPredictor == NO_PREDICTOR {
		t.Classification, err = t.getMajorityVote()
		return nil
	} else {
		err = t.splitNode(bestPredictor, *bestIndex)
		if err != nil {
			return err
		}
	}

	// automatically continue on child nodes
	if auto {
		if t.left != nil {
			t.left.Expand(true)
		}
		if t.right != nil {
			t.right.Expand(true)
		}
	}

	return
}

// Returns 0 iff at least half of the observations in this node have target value 0; 1 otherwise.
func (t *DecisionTree) getMajorityVote() (bestVal Value, err error) {
	if len(t.Observations) == 0 {
		return nil, errors.New("Cannot vote on an empty node!")
	}

	votes := map[Value]int{}
	bestVal, bestVotes := (*t.Observations[0])[t.Options.TargetAttribute], 0

	for _, obs := range t.Observations {
		thisVal := (*obs)[t.Options.TargetAttribute]
		votes[thisVal]++

		obsVotes := votes[thisVal]
		if obsVotes > bestVotes {
			bestVotes, bestVal = obsVotes, thisVal
		}
	}
	return
}

// Taking two arguments, the predictor and index, splits the node into two subtrees, left one containing
// all observations from the 1st smallest up to the index-th smallest, sorted according to the predictor.
// The right node contains the rest of the observations.
func (t *DecisionTree) splitNode(predictor string, index int) error {
	if index < 0 || index >= len(t.Observations) {
		return errors.New("Invalid split index.")
	}

	// Remember split information
	t.sortByPredictor(predictor)
	t.SplitPredictor = &predictor
	t.SplitValue = (*t.Observations[index])[predictor]

	// Set up the child nodes
	t.setLeft(new(DecisionTree))
	t.setRight(new(DecisionTree))
	t.left.Observations = t.Observations[:index]
	t.right.Observations = t.Observations[index:]
	return nil
}

// Returns the classification for a new observation o.
func (t *DecisionTree) Classify(o *Observation) (val Value, err error) {
	feature, val, classification := t.GetRule()

	if t.IsLeaf() {
		return classification, err
	}

	//Traverse the tree
	if isLess, err := _lt((*o)[feature], val); isLess && err == nil {
		return t.left.Classify(o)
	} else if err != nil {
		return nil, err
	}

	return t.right.Classify(o)
}

// Returns a tuple describing the split rule for this node.
// Format for leaf nodes: <NO_PREDICTOR>,<NO_FLOAT>,<classification at node>
// Format for internal nodes: <split predictor>,<split value>,<NO_CLASSIFICATION>
func (t *DecisionTree) GetRule() (predictor string, splitValue Value, classification Value) {
	if t.IsLeaf() {
		return NO_PREDICTOR, nil, t.Classification
	} else {
		return *t.SplitPredictor, t.SplitValue, nil
	}
}
