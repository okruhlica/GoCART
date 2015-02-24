package cart

type NodeValue struct {
	dataType  [1]string
	boolVal   *bool
	floatVal  *float64
	stringVal *string
}

type CartTree struct {
	Left, Right, Up *CartTree
	Good, Bad       int
	Vals            []*NodeValue
}

func IsLeaf(t *CartTree) bool {
	return t.Left == nil && t.Right == nil
}

func AddLeft(parent *CartTree, child *CartTree) {
	child.Up = parent
	parent.Left = child
}

func AddRight(parent *CartTree, child *CartTree) {
	child.Up = parent
	parent.Right = child
}
