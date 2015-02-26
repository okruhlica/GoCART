package rtree

type GrowOptions struct {
	minSplitSize int
	maxSplitGini float64
	maxDepth     int
}
