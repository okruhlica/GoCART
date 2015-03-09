package decision_tree

type Options struct {
	MinSplitSize     int
	MaxSplitImpurity float64
	MaxDepth         int
	SplitStrategy    AbstractPurityMetric
	TargetAttribute  string
	Predictors       *[]string
}
