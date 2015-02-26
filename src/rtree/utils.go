package rtree

import "strings"
import "strconv"
import "fmt"

// Returns a list of leaf nodes for the given tree.
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

// Returns the list of predictors used in the given tree for decision-making.
func (t *rtree) GetUsedPredictors() []string {
	if t.SplitPredictor == nil || *t.SplitPredictor == NO_PREDICTOR {
		return []string{}
	}

	out := map[string]int{*t.SplitPredictor: 1}
	if t.Left != nil {
		for _, v := range t.Left.GetUsedPredictors() {
			out[v] = 1
		}
	}
	if t.Right != nil {
		for _, v := range t.Right.GetUsedPredictors() {
			out[v] = 1
		}
	}

	keys := make([]string, 0, len(out))
	for k := range out {
		keys = append(keys, k)
	}
	return keys
}

// Prints the tree to stdout.
// depth - for formatting purposes, use 0 on invocation
// verbose - if true, it prints the same information, moreover printing the list of observations for each node
func (t *rtree) PrintTree(depth int, verbose bool) string {
	if t != nil {
		splitPredictor, splitVal := NO_PREDICTOR, -1.0
		if t.SplitValue != nil && t.SplitPredictor != nil {
			splitVal = t.SplitValue.Float
			splitPredictor = *t.SplitPredictor
		}
		if verbose {
			fmt.Printf("%s (rule: %s < %f, gini: %f, f(y)=%d)[%d observations:[%s]]\n", strings.Repeat("--|", depth+1),
				splitPredictor,
				splitVal,
				t.Impurity,
				t.Classification,
				len(t.Observations),
				SerializeObservations(t.Observations))
		} else {
			fmt.Printf("%s (rule: %s < %f, gini: %f, f(y)=%d)\n", strings.Repeat("-", depth*3+1),
				splitPredictor,
				splitVal,
				t.Impurity,
				t.Classification)
		}
		return t.Left.PrintTree(depth+1, verbose) + t.Right.PrintTree(depth+1, verbose)
	}
	return ""
}

func SerializeObservations(observations []*Observation) string {
	out := ""
	for _, obs := range observations {
		out += (",[y=" + strconv.FormatFloat((*obs)[TARGET_KEY].Float, 'f', 6, 64)) +
			(",id=" + strconv.FormatFloat((*obs)["__id"].Float, 'f', 6, 64) + "]")
	}
	return out
}
