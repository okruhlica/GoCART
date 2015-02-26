package rtree

import "strings"
import "strconv"
import "fmt"

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
				serializeObservations(t.Observations))
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

func serializeObservations(observations []*Observation) string {
	out := ""
	for _, obs := range observations {
		out += (",[y=" + strconv.FormatFloat((*obs)[TARGET_KEY].Float, 'f', 6, 64)) +
			(",id=" + strconv.FormatFloat((*obs)["__id"].Float, 'f', 6, 64) + "]")
	}
	return out
}
