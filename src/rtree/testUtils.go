package rtree

import "strconv"

func stofval(s string) *Value {
	float, _ := strconv.ParseFloat(s, 64)
	return &Value{float}
}
