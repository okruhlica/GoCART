package rtree

type Value struct {
	Float float64
}

const TARGET_KEY = "__target"

type Observation map[string]*Value
