package rtree

type GrowOptions struct {
	MinSplitSize  int
	MaxSplitGini  float64
	MaxDepth      int
	SplitStrategy PurityMetricStrategy
}
