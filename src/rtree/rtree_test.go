package rtree

import "testing"

import "sort"

import "math"
import "os"
import "fmt"
import "encoding/csv"
import "strings"
import "strconv"

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

func getSettings(name string) *GrowOptions {
	switch name {
	case "supergrow":
		return &GrowOptions{2, 0.0, 10, GiniPurity}
	case "shallow":
		return &GrowOptions{25, 0.15, 10, GiniPurity}
	}
	return getSettings("supergrow")
}

func Test_GetRule(tst *testing.T) {
	observations := prepareTestObservations([]string{})
	t := new(rtree)
	t.InitRoot(getSettings("supergrow"), *observations).Expand(true)

	//test case 1
	predictor, val, classif := t.GetRule()
	if predictor == "feature3" && val == 15.0 && classif == NO_CLASSIFICATION {
		tst.Log("[cart/Test_GetRule] Case 1 passed.")
	} else {
		tst.Errorf("[cart/Test_GetRule] Case 1 failed. Expected ('feature3', 15.0, %d), got (%s,%f,%d)", NO_CLASSIFICATION, predictor, val, classif)
	}

	//test case 2
	predictor, val, classif = t.Left.GetRule()
	if predictor == NO_PREDICTOR && val == NO_FLOAT && classif == 1 {
		tst.Log("[cart/Test_GetRule] Case 2 passed.")
	} else {
		tst.Errorf("[cart/Test_GetRule] Case 2 failed. Expected ('%s', %f, 1), got (%s,%f,%d)", NO_PREDICTOR, NO_FLOAT, predictor, val, classif)
	}
}

func prepareTestObservations(excludeFeatures []string) *[]*Observation {
	val1 := Value{-0.5}
	val2 := Value{7.47}
	val3 := Value{10.55}
	val4 := Value{15.00}
	val5 := Value{20.00}
	valt0 := Value{0.0}
	valt1 := Value{1.0}

	obs1 := Observation{
		"feature1": &val1,
		"feature2": &val2,
		"feature3": &val1,
		"__id":     &Value{1.0},
		TARGET_KEY: &valt1,
	}

	obs2 := Observation{
		"feature1": &val2,
		"feature2": &val3,
		"feature3": &val4,
		"__id":     &Value{2.0},
		TARGET_KEY: &valt0,
	}

	obs3 := Observation{
		"feature1": &val3,
		"feature2": &val1,
		"feature3": &val5,
		"__id":     &Value{3.0},
		TARGET_KEY: &valt0,
	}

	obs4 := Observation{
		"feature1": &val4,
		"feature2": &val5,
		"feature3": &val2,
		"__id":     &Value{4.0},
		TARGET_KEY: &valt1,
	}

	obs5 := Observation{
		"feature1": &val5,
		"feature2": &val4,
		"feature3": &val3,
		"__id":     &Value{5.0},
		TARGET_KEY: &valt1,
	}
	if len(excludeFeatures) > 0 {
		for _, feature := range excludeFeatures {
			delete(obs1, feature)
			delete(obs2, feature)
			delete(obs3, feature)
			delete(obs4, feature)
			delete(obs5, feature)
		}
	}
	observations := []*Observation{&obs1, &obs2, &obs3, &obs4, &obs5}
	return &observations
}

func Test_Gini(tst *testing.T) {
	observations := *prepareTestObservations([]string{})
	assertGini(tst, "[:0]", gini(observations[:0]), 0.0)
	assertGini(tst, "[:1]", gini(observations[:1]), 0.0)
	assertGini(tst, "[:2]", gini(observations[:2]), 0.5)
	assertGini(tst, "[:3]", gini(observations[:3]), 0.44444)
	assertGini(tst, "[:4]", gini(observations[:4]), 0.5)
	assertGini(tst, "[:4]", gini(observations[:5]), 0.48)
}

func assertGini(tst *testing.T, testLabel string, err float64, exp_err float64) {
	if math.Abs(err-exp_err) > 0.001 {
		tst.Errorf("Gini impurity test (%s) failed. Value got is %f, expected %f.", testLabel, err, exp_err)
	} else {
		tst.Logf("Gini index test (%s) passed.", testLabel)
	}
}

func Test_Classify(tst *testing.T) {
	t := new(rtree)
	observations := prepareTestObservations([]string{})
	t.Observations = *observations
	t.InitNode(getSettings("supergrow"), 0)
	t.Expand(true)

	obsNew := Observation{
		"feature1": &Value{4.7},
		"feature2": &Value{4.7},
		"feature3": &Value{40.7},
	}
	outcome := t.Classify(&obsNew)

	if outcome == 0 {
		tst.Logf("Classification test passed (classification = 1).")
	} else {
		tst.Errorf("Classification test failed (expected = %d, was %d).", 1-outcome, outcome)
	}
}

func Test_FindBestSplit(tst *testing.T) {
	testBestSplitCase(tst, "features={1,2,3}", []string{}, "feature3", 3, 0.0)
	testBestSplitCase(tst, "features={1,2}", []string{"feature3"}, "feature1", 3, 0.222222)
	testBestSplitCase(tst, "features={1}", []string{"feature2", "feature3"}, "feature1", 3, 0.222222)
	testBestSplitCase(tst, "features={}", []string{"feature1", "feature2", "feature3"}, NO_PREDICTOR, NO_INDEX, NO_FLOAT)
	testBestSplitCase(tst, "features={1}", []string{"feature1", "feature3"}, "feature2", 3, 0.222222)
}

func testBestSplitCase(tst *testing.T, testName string, excludeFeatures []string, expectedPredictor string, expectedIndex int, expectedGini float64) {
	t := new(rtree)

	observations := prepareTestObservations(excludeFeatures)
	t.Observations = *observations
	t.InitNode(getSettings("supergrow"), 0)
	predictor, index, gini := t.FindBestSplit()

	if predictor == expectedPredictor && index == expectedIndex && math.Abs(gini-expectedGini) < 0.001 {
		tst.Logf("Find best split test '%s' passed.", testName)
	} else {
		tst.Errorf("Find best split test '%s' failed. Expected (%s,%d,%f) but got (%s,%d,%f).", testName, expectedPredictor, expectedIndex, expectedGini, predictor, index, gini)
	}
}

func Test_Split(tst *testing.T) {
	t := new(rtree)
	observations := prepareTestObservations([]string{})
	t.Observations = *observations
	t.Split(2)

	if t.Left.Observations[0] == t.Observations[0] &&
		t.Left.Observations[1] == t.Observations[1] &&
		t.Right.Observations[0] == t.Observations[2] &&
		t.Right.Observations[1] == t.Observations[3] &&
		t.Right.Observations[2] == t.Observations[4] &&
		len(t.Left.Observations) == 2 &&
		len(t.Right.Observations) == 3 {
		tst.Logf("Test split node passed.")
	} else {
		tst.Errorf("Test split node failed. Expected %p, got %p. Tree is %s", t.Observations[0], t.Left.Observations[0] /*, t.PrintTree(0, true)*/)
	}
}

func Test_ExpandNode(tst *testing.T) {
	t := new(rtree)
	observations := prepareTestObservations([]string{})
	t.Observations = *observations
	t.InitNode(getSettings("supergrow"), 0)
	t.Expand(true)

	if len(t.Left.Observations) == 3 &&
		len(t.Right.Observations) == 2 {
		//		tst.Logf("Test expand node passed with tree %s", t.PrintTree(0, true))
	} else {
		tst.Errorf("Test expand node failed. Expected to split into (%d,%d) nodes, got (%d,%d). Tree is %s", 3, 2, len(t.Left.Observations), len(t.Right.Observations) /*, t.PrintTree(0, true)*/)
	}
}

func BenchmarkExpandNode(b *testing.B) {
	observations := prepareTestObservations([]string{})
	for n := 0; n < b.N; n++ {
		t := new(rtree)
		t.Observations = *observations
		t.InitNode(getSettings("supergrow"), 0)
		t.Expand(true)
	}
}

func Test_GetMajorityVote(tst *testing.T) {
	t := new(rtree)

	//Test case 1
	observations := prepareTestObservations([]string{})
	t.Observations = *observations
	t.InitNode(getSettings("supergrow"), 0)
	if t.GetMajorityVote() == 1 {
		tst.Logf("Test majority vote passed.")
	} else {
		tst.Errorf("Test majority vote failed. Expected 1, got %d", t.GetMajorityVote())
	}

	//Test case 2
	t.Observations = (*observations)[:3]
	t.InitNode(getSettings("supergrow"), 0)
	if t.GetMajorityVote() == 0 {
		tst.Logf("Test majority vote passed.")
	} else {
		tst.Errorf("Test majority vote failed. Expected 0, got %d", t.GetMajorityVote())
	}
}

func Test_SortFunc(tst *testing.T) {
	val1 := Value{-0.5}
	val2 := Value{7.47}
	val3 := Value{10.55}

	obs1 := Observation{
		"feature1": &val1,
		"feature2": &val2,
	}

	obs2 := Observation{
		"feature1": &val2,
		"feature2": &val3,
	}

	obs3 := Observation{
		"feature1": &val3,
		"feature2": &val1,
	}

	observations := []*Observation{&obs1, &obs2, &obs3}
	sortFunc := ByPredictorValueFloat{"feature2", &observations}
	sort.Sort(sortFunc)
	if (*observations[0])["feature2"].Float == -0.5 &&
		(*observations[1])["feature2"].Float == 7.47 &&
		(*observations[2])["feature2"].Float == 10.55 {
		tst.Log("[cart/Test_SortFunc] Sort func test 1 OK.")
	} else {
		tst.Log((*observations[0])["feature2"].Float, obs2["feature2"].Float)
		tst.Error("[cart/Test_SortFunc] Sort func test 1 failed.")
	}

	sortFunc = ByPredictorValueFloat{"feature1", &observations}
	sort.Sort(sortFunc)
	if (*observations[0])["feature1"].Float == -0.5 &&
		(*observations[1])["feature1"].Float == 7.47 &&
		(*observations[2])["feature1"].Float == 10.55 {
		tst.Log("[cart/Test_SortFunc] Sort func test 2 OK.")
	} else {
		tst.Log((*observations[0])["feature1"].Float, obs2["feature1"].Float)
		tst.Error("[cart/Test_SortFunc] Sort func test 2 failed.")
	}
}

func Test_CsvDataSet(tst *testing.T) {
	t := new(rtree)
	obs := loadCsvDataset("testData/data2.csv", 90000, 0)
	t.Observations = *obs
	t.InitNode(getSettings("shallow"), 0)
	t.Expand(true)

	obss := loadCsvDataset("testData/data2.csv", 10000, 90000)
	successes := 0
	for _, obs := range *obss {
		if float64(t.Classify(obs)) == (*(*obs)[TARGET_KEY]).Float {
			successes++
		}
	}

	t.PrintTree(0, false)
	tst.Errorf("Success rate is %d/1000", successes)
}

func loadCsvDataset(path string, takeN int, skipN int) *[]*Observation {

	csvfile, err := os.Open(path)

	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	defer csvfile.Close()

	reader := csv.NewReader(csvfile)

	reader.FieldsPerRecord = -1

	rawCSVdata, err := reader.ReadAll()

	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// sanity check, display to standard output
	observations := []*Observation{}
	for i, each := range rawCSVdata {

		if i < skipN {
			continue
		} else if i > skipN+takeN {
			break
		}

		//		fmt.Printf("id : %s, __target: %s, x: %s, y: %s, z: %s\n", each[0], each[1], each[2], each[3], each[4])
		newObs := Observation{
			TARGET_KEY: stofval(each[1]),
			"__id":     &Value{float64(i)},
		}

		for i, e := range each[2:] {
			newObs["attr_"+strconv.Itoa(i+2)] = stofval(e)
		}
		observations = append(observations, &newObs)
	}

	return &observations
}

func Test_FindPredictorSplit(tst *testing.T) {
	t := new(rtree)
	observations := prepareTestObservations([]string{})
	t.InitRoot(getSettings("supergrow"), *observations)
	splitIdx, splitGini := t.BestSplitWithPredictor("feature1")

	if splitIdx == 3 && math.Abs(splitGini-0.222222) < 0.001 {
		tst.Log("Find split test passed.")
	} else {
		tst.Errorf("Find split test failed, expected (%f,%d) but got (%f,%d).", 0.222222, 3, splitGini, splitIdx)
	}
}

func Test_CummulativeGoodCounts(tst *testing.T) {
	goods := *calculateCummulativeGoodSlice(prepareTestObservations([]string{}))

	if goods[0] == 0 && goods[1] == 1 && goods[2] == 1 && goods[3] == 1 && goods[4] == 2 && goods[5] == 3 && goods[len(goods)-1] == 3 {
		tst.Log("Cummulative good target calculation passed.")
	} else {
		tst.Errorf("Cummulative good target calculation failed. Expected[0,1,1,1,2,3], but got [%d,%d,%d,%d,%d].", goods[0], goods[1], goods[2], goods[3], goods[4], goods[5])
	}
}

func Test_GetUsedPredictors(tst *testing.T) {
	t := new(rtree)

	// Case 1
	t.InitRoot(getSettings("supergrow"), *prepareTestObservations([]string{})).Expand(true)
	predictors := t.GetUsedPredictors()
	if len(predictors) == 1 && predictors[0] == "feature3" {
		tst.Log("[test/GetUsedPredictors] Case 1 passed.")
	} else {
		tst.Errorf("[test/GetUsedPredictors] Case 1 failed.")
	}

	// Case 2
	t.InitRoot(getSettings("supergrow"), *prepareTestObservations([]string{"feature3"})).Expand(true)
	predictors = t.GetUsedPredictors()
	if len(predictors) == 1 && predictors[0] == "feature1" {
		tst.Log("[test/GetUsedPredictors] Case 2 passed.")
	} else {
		tst.Errorf("[test/GetUsedPredictors] Case 2 failed.")
	}
}
