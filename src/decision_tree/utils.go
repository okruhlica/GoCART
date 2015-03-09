package decision_tree

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Returns true iff the given attribute name is a valid for splitting upon.
// Only attributes that do not start with two underscores can be used as predictors.
func (t *DecisionTree) isPredictor(predictor string) bool {
	for _, p := range *t.Options.Predictors {
		if predictor == p {
			return true
		}
	}
	return false
}

// Returns true iff this node has no children
func (t *DecisionTree) IsLeaf() bool {
	return t.left == nil && t.right == nil
}

// Returns a list of leaf nodes for the given tree.
func (t *DecisionTree) GetLeaves() []*DecisionTree {
	if t.IsLeaf() {
		return []*DecisionTree{t}
	}

	leaves := []*DecisionTree{}
	if t.left != nil {
		leaves = t.left.GetLeaves()
	}

	if t.right != nil {
		return append(leaves, t.right.GetLeaves()...)
	}
	return leaves
}

// Returns the list of predictors used in the given tree for decision-making.
func (t *DecisionTree) GetUsedPredictors() []string {
	if t.SplitPredictor == nil || *t.SplitPredictor == NO_PREDICTOR {
		return []string{}
	}

	out := map[string]int{*t.SplitPredictor: 1}
	if t.left != nil {
		for _, v := range t.left.GetUsedPredictors() {
			out[v] = 1
		}
	}
	if t.right != nil {
		for _, v := range t.right.GetUsedPredictors() {
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
func (t *DecisionTree) PrintTree1(depth int, verbose bool) string {
	return t.PrintTree(depth, 10000, verbose)
}

// Prints the tree to stdout.
// depth - for formatting purposes, use 0 on invocation
// verbose - if true, it prints the same information, moreover printing the list of observations for each node
func (t *DecisionTree) PrintTree(depth int, maxDepth int, verbose bool) string {
	if depth > maxDepth {
		return ""
	}

	if t != nil {
		if t.IsLeaf() {
			classif, _ := _str(t.Classification)
			if verbose {
				fmt.Printf("%s Classification=%s [%d observations: [%s]]\n\n", strings.Repeat("--|", depth+1), classif, len(t.Observations), SerializeObservations(t.Observations, t.Options.TargetAttribute))
			} else {
				fmt.Printf("%s Classification=%s [%d observations]\n\n", strings.Repeat("--|", depth+1), classif, len(t.Observations))
			}
		} else {
			splitPredictor := NO_PREDICTOR
			var splitVal interface{}
			splitVal, _ = _str(t.SplitValue)
			splitPredictor = *t.SplitPredictor
			if verbose {
				fmt.Printf("%s (rule: %s < %s)[%d observations:[%s]]\n\n", strings.Repeat("--|", depth+1),
					splitPredictor,
					splitVal,
					len(t.Observations),
					SerializeObservations(t.Observations, t.Options.TargetAttribute))
			} else {
				fmt.Printf("%s (rule: %s < %s)\n\n", strings.Repeat("--|", depth+1),
					splitPredictor,
					splitVal)
			}
			return t.left.PrintTree(depth+1, maxDepth, verbose) + t.right.PrintTree(depth+1, maxDepth, verbose)
		}
	}
	return ""
}

// === Serialization

func SerializeObservations(observations []*Observation, targetKey string) (out string) {
	for _, obs := range observations {
		for attr, val := range *obs {
			sVal, _ := _str(val)
			out += attr + "=" + sVal + ","
		}
		out += "];"
	}
	return
}

type serializedTree struct {
	SplitOn        *string         `json:"splitOn,omitempty"`
	SplitValue     Value           `json:"splitValue,omitempty"`
	IfLeq          *serializedTree `json:"ifLeq,omitempty"`
	IfGt           *serializedTree `json:"ifGt,omitempty"`
	Classification Value           `json:"classification,omitempty"`
}

func (t *DecisionTree) GetSerializedModel() string {
	st := t.constructSerializedModel()
	b, _ := json.Marshal(st)
	return string(b)
}

func (t *DecisionTree) constructSerializedModel() *serializedTree {
	m := &serializedTree{}

	if t.IsLeaf() {
		m.Classification = t.Classification
	} else {
		m.SplitOn = t.SplitPredictor
		v := t.SplitValue.(float64)
		m.SplitValue = &v
	}
	if t.left != nil {
		m.IfLeq = t.left.constructSerializedModel()
	}
	if t.right != nil {
		m.IfGt = t.right.constructSerializedModel()
	}
	return m
}

// ====== operator wrappers ======

func _lt(l interface{}, r interface{}) (retVal bool, err error) {
	switch lv := l.(type) {
	case float32:
		if rv, f := r.(float32); f {
			retVal = lv < rv
			return
		}
	case float64:
		if rv, f := r.(float64); f {
			retVal = lv < rv
			return
		}
	case int:
		if rv, f := r.(int); f {
			retVal = lv < rv
			return
		}
	case string:
		if rv, f := r.(string); f {
			retVal = lv < rv
			return
		}
	}
	return false, errors.New("Uncomparable or unsupported types.")
}

func _eq(l interface{}, r interface{}) (retVal bool, err error) {
	switch lv := l.(type) {
	case float32:
		if rv, f := r.(float32); f {
			retVal = lv == rv
			return
		}
	case float64:
		if rv, f := r.(float64); f {
			retVal = lv == rv
			return
		}
	case int:
		if rv, f := r.(int); f {
			retVal = lv == rv
			return
		}
	case string:
		if rv, f := r.(string); f {
			retVal = lv == rv
			return
		}
	}
	return false, errors.New("Uncomparable or unsupported types.")
}

func _gt(l interface{}, r interface{}) (retVal bool, err error) {
	switch lv := l.(type) {
	case float32:
		if rv, f := r.(float32); f {
			retVal = lv > rv
			return
		}
	case float64:
		if rv, f := r.(float64); f {
			retVal = lv > rv
			return
		}
	case int:
		if rv, f := r.(int); f {
			retVal = lv > rv
			return
		}
	case string:
		if rv, f := r.(string); f {
			retVal = lv > rv
			return
		}
	}
	return false, errors.New("Uncomparable or unsupported types.")
}

func _str(v interface{}) (str string, err error) {
	switch vv := v.(type) {
	case float32:
		return strconv.FormatFloat(float64(vv), 'f', 6, 64), err
	case float64:
		return strconv.FormatFloat(vv, 'f', 6, 64), err
	case int:
		return strconv.Itoa(vv), err
	case string:
		return vv, err
	}
	return "", errors.New("Uncomparable or unsupported types.")
}
