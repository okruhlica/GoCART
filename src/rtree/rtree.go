package rtree

type NodeValue struct {
	floatVal float64
}

type rtree struct {
	Left, Right, Up *rtree
	Value           []*NodeValue
	good, bad       int
}

func (t *rtree) IsLeaf() bool {
	return t.Left == nil && t.Right == nil
}

func (t *rtree) SetLeft(child *rtree) {
	child.Up = t
	t.Left = child
}

func (t *rtree) SetRight(child *rtree) {
	child.Up = t
	t.Right = child
}

func (t *rtree) GetLeaves() []*rtree {
	if t.IsLeaf() {
		return []*rtree{t}
	}

	leaves := []*rtree{}
	if t.Left != nil {
		leaves = t.Left.GetLeaves()
	}

	if t.Right != nil {
		rightLeaves := t.Right.GetLeaves()
		return append(leaves, rightLeaves...)
		return leaves
	}
	return leaves
}
