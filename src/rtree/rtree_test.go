package rtree

import "testing"

func cmpSlices(X, Y []*rtree) bool {
	for _, y := range Y {
		found := false
		for _, x := range X {
			if x == y {
				found = true
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func Test_IsLeaf(tst *testing.T) {
	t := new(rtree)
	t1 := new(rtree)
	t2 := new(rtree)

	if t.IsLeaf() {
		tst.Log("[cart/Test_IsLeaf] Leaf test 1 passed (isLeaf=true)")
	} else {
		tst.Error("[cart/Test_IsLeaf] Leaf test 1 failed (isLeaf=true)")
	}

	t.SetLeft(t1)
	if t.Left != t1 {
		tst.Error("[cart/Test_IsLeaf] Left child was not added properly.")
	}

	if !t.IsLeaf() {
		tst.Log("[cart/Test_IsLeaf] Leaf test 2 passed (isLeaf=false)")
	} else {
		tst.Error("[cart/Test_IsLeaf] Leaf test 2 failed (isLeaf=false)")
	}

	t.SetLeft(t2)

	if !t.IsLeaf() {
		tst.Log("[cart/Test_IsLeaf] Leaf test 3 passed (isLeaf=false)")
	} else {
		tst.Error("[cart/Test_IsLeaf] Leaf test 3 failed (isLeaf=false)")
	}
}

func Test_GetLeaves(tst *testing.T) {
	t := new(rtree)
	t1 := new(rtree)
	t2 := new(rtree)
	t3 := new(rtree)

	if cmpSlices(t.GetLeaves(), []*rtree{t}) {
		tst.Log("[cart/Test_GetLeaves] Get leaves test 1 passed (GetLeaves=[])")
	} else {
		tst.Error("[cart/Test_GetLeaves] Get leaves test 1 failed (GetLeaves=[])")
	}

	t.SetLeft(t1)
	if cmpSlices(t.GetLeaves(), []*rtree{t1}) {
		tst.Log("[cart/Test_GetLeaves] Get leaves test 2 passed (GetLeaves=[t1])")
	} else {
		tst.Error("[cart/Test_GetLeaves] Get leaves test 2 failed (GetLeaves=[t1])")
	}

	t.SetRight(t2)
	t1.SetRight(t3)

	if cmpSlices(t.GetLeaves(), []*rtree{t2, t3}) {
		tst.Log("[cart/Test_GetLeaves] Get leaves test 3 passed (GetLeaves=[t2,t3])")
	} else {
		tst.Error("[cart/Test_GetLeaves] Get leaves test 3 failed (GetLeaves=[t2,t3])")
	}
}
