package decision_tree

import "testing"
import "sort"
import "math"
import "os"
import "fmt"
import "encoding/csv"
import "strconv"

const TARGET_KEY = "__target"

func cmpSlices(X, Y []*DecisionTree) bool {
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
	t := new(DecisionTree)
	t1 := new(DecisionTree)
	t2 := new(DecisionTree)
	t.InitRoot(getSettings("supergrow", "__target"), []*Observation{})
	t1.InitRoot(getSettings("supergrow", "__target"), []*Observation{})
	t2.InitRoot(getSettings("supergrow", "__target"), []*Observation{})

	if t.IsLeaf() {
		tst.Log("[decision_tree/Test_IsLeaf] Leaf test 1 passed (isLeaf=true)")
	} else {
		tst.Error("[decision_tree/Test_IsLeaf] Leaf test 1 failed (isLeaf=true)")
	}

	t.setLeft(t1)
	if t.left != t1 {
		tst.Error("[decision_tree/Test_IsLeaf] Left child was not added properly.")
	}

	if !t.IsLeaf() {
		tst.Log("[decision_tree/Test_IsLeaf] Leaf test 2 passed (isLeaf=false)")
	} else {
		tst.Error("[decision_tree/Test_IsLeaf] Leaf test 2 failed (isLeaf=false)")
	}

	t.setLeft(t2)

	if !t.IsLeaf() {
		tst.Log("[decision_tree/Test_IsLeaf] Leaf test 3 passed (isLeaf=false)")
	} else {
		tst.Error("[decision_tree/Test_IsLeaf] Leaf test 3 failed (isLeaf=false)")
	}
}

func Test_GetLeaves(tst *testing.T) {
	t := new(DecisionTree)
	t1 := new(DecisionTree)
	t2 := new(DecisionTree)
	t3 := new(DecisionTree)

	t.InitRoot(getSettings("supergrow", "__target"), []*Observation{})
	t1.InitRoot(getSettings("supergrow", "__target"), []*Observation{})
	t2.InitRoot(getSettings("supergrow", "__target"), []*Observation{})
	t3.InitRoot(getSettings("supergrow", "__target"), []*Observation{})

	if cmpSlices(t.GetLeaves(), []*DecisionTree{t}) {
		tst.Log("[decision_tree/Test_GetLeaves] Get leaves test 1 passed (GetLeaves=[])")
	} else {
		tst.Error("[decision_tree/Test_GetLeaves] Get leaves test 1 failed (GetLeaves=[])")
	}

	t.setLeft(t1)
	if cmpSlices(t.GetLeaves(), []*DecisionTree{t1}) {
		tst.Log("[decision_tree/Test_GetLeaves] Get leaves test 2 passed (GetLeaves=[t1])")
	} else {
		tst.Error("[decision_tree/Test_GetLeaves] Get leaves test 2 failed (GetLeaves=[t1])")
	}

	t.setRight(t2)
	t1.setRight(t3)

	if cmpSlices(t.GetLeaves(), []*DecisionTree{t2, t3}) {
		tst.Log("[decision_tree/Test_GetLeaves] Get leaves test 3 passed (GetLeaves=[t2,t3])")
	} else {
		tst.Error("[decision_tree/Test_GetLeaves] Get leaves test 3 failed (GetLeaves=[t2,t3])")
	}
}

func getSettings(name string, targetKey string) *Options {
	giniStrategy := GiniPurity{}
	switch name {
	case "supergrow":
		return &Options{2, 0.0, 10, giniStrategy, targetKey, &[]string{"feature1", "feature2", "feature3"}}
	case "supergrow-nofeatures":
		return &Options{2, 0.0, 10, giniStrategy, targetKey, &[]string{}}
	case "shallow":
		return &Options{25, 0.15, 10, giniStrategy, targetKey, &[]string{"feature1", "feature2", "feature3"}}
	}
	return getSettings("supergrow", targetKey)
}

func prepareTestObservations(excludeFeatures []string) []*Observation {
	var val1, val2, val3, val4, val5, valt0, valt1 interface{}
	val1 = -0.5
	val2 = 7.47
	val3 = 10.55
	val4 = 15.00
	val5 = 20.00
	valt0 = 0.0
	valt1 = 1.0

	obs1 := Observation{
		"feature1": val1,
		"feature2": val2,
		"feature3": val1,
		"__id":     "Example #1",
		TARGET_KEY: valt1,
	}

	obs2 := Observation{
		"feature1": val2,
		"feature2": val3,
		"feature3": val4,
		"__id":     "Example #2",
		TARGET_KEY: valt0,
	}

	obs3 := Observation{
		"feature1": val3,
		"feature2": val1,
		"feature3": val5,
		"__id":     "Example #3",
		TARGET_KEY: valt0,
	}

	obs4 := Observation{
		"feature1": val4,
		"feature2": val5,
		"feature3": val2,
		"__id":     "Example #4",
		TARGET_KEY: valt1,
	}

	obs5 := Observation{
		"feature1": val5,
		"feature2": val4,
		"feature3": val3,
		"__id":     5.0,
		TARGET_KEY: valt1,
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
	return observations
}

// ================================= TESTS START HERE =================================

func Test_Gini(tst *testing.T) {
	gini := new(GiniPurity)

	observations := prepareTestObservations([]string{})

	pur0, _ := gini.SlicePurity(observations[:0], "__target")
	pur1, _ := gini.SlicePurity(observations[:1], "__target")
	pur2, _ := gini.SlicePurity(observations[:2], "__target")
	pur3, _ := gini.SlicePurity(observations[:3], "__target")
	pur4, _ := gini.SlicePurity(observations[:4], "__target")
	pur5, _ := gini.SlicePurity(observations[:5], "__target")

	assertGini(tst, "[:0]", pur0, 0.0)
	assertGini(tst, "[:1]", pur1, 0.0)
	assertGini(tst, "[:2]", pur2, 0.5)
	assertGini(tst, "[:3]", pur3, 0.44444)
	assertGini(tst, "[:4]", pur4, 0.5)
	assertGini(tst, "[:5]", pur5, 0.48)
}

func assertGini(tst *testing.T, testLabel string, gotGini float64, expectedGini float64) {
	if math.Abs(gotGini-expectedGini) > 0.001 {
		tst.Errorf("Gini impurity test (%s) failed. Value got is %f, expected %f.", testLabel, gotGini, expectedGini)
	} else {
		tst.Logf("Gini index test (%s) passed.", testLabel)
	}
}

func Test_FindPredictorSplit(tst *testing.T) {
	testBestPredictorSplitCase(tst, "{1,2,3}/f1", []string{}, "feature1", 3, 0.22222)

	testBestPredictorSplitCase(tst, "{1,2,3}/f1", []string{}, "feature3", 3, 0.0)
	testBestPredictorSplitCase(tst, "{1,2,3}/f2", []string{"feature1", "feature3"}, "feature2", 3, 0.222222)
	testBestPredictorSplitCase(tst, "{1,2,3}/f1_2", []string{"feature2", "feature3"}, "feature1", 3, 0.222222)
	testBestPredictorSplitCase(tst, "{1,2,3}/f3", []string{"feature1", "feature2"}, "feature3", 3, 0.0)
}

func testBestPredictorSplitCase(tst *testing.T, testName string, excludeFeatures []string, predictor string, expectedIndex int, expectedGini float64) {
	t := new(DecisionTree)

	observations := prepareTestObservations(excludeFeatures)
	t.InitRoot(getSettings("supergrow", "__target"), observations)
	index, gini, _ := t.bestSplitWithPredictor(predictor)

	if *index == expectedIndex && math.Abs(*gini-expectedGini) < 0.001 {
		tst.Logf("Find best predictor split test '%s' passed.", testName)
	} else {
		tst.Errorf("Find best predictor split test '%s' failed. Expected (%d,%f) but got (%d,%f).", testName, expectedIndex, expectedGini, *index, *gini)
	}
}

func Test_FindBestSplit(tst *testing.T) {
	three, zero, pt2 := 3, 0.0, 0.222222

	testBestSplitCase(tst, "features={1,2,3}", "supergrow", []string{}, "feature3", &three, &zero)
	testBestSplitCase(tst, "features={1,2}", "supergrow", []string{"feature3"}, "feature1", &three, &pt2)
	testBestSplitCase(tst, "features={1}", "supergrow", []string{"feature2", "feature3"}, "feature1", &three, &pt2)
	testBestSplitCase(tst, "features={}", "supergrow-nofeatures", []string{"feature1", "feature2", "feature3"}, NO_PREDICTOR, nil, nil)
}

func testBestSplitCase(tst *testing.T, testName string, settings string, excludeFeatures []string, expPredictor string, expIndex *int, expImp *float64) {
	t := new(DecisionTree)

	observations := prepareTestObservations(excludeFeatures)
	err := t.InitRoot(getSettings(settings, "__target"), observations)
	if err != nil {
		tst.Errorf("Failed with %s", err.Error())
	}
	var (
		gotPredictor string
		gotIndex     *int
		gotImp       *float64
	)
	gotPredictor, gotIndex, gotImp, _ = t.FindBestSplit()

	if gotPredictor == expPredictor &&
		((expIndex == nil && gotIndex == nil) || *gotIndex == *expIndex) &&
		(expImp == nil && gotImp == nil) || (math.Abs(*gotImp-*expImp) < 0.001) {
		tst.Logf("Find best split test '%s' passed.", testName)
	} else {
		if expIndex == nil || expImp == nil {
			tst.Errorf("Find best split test '%s' failed. Expected (%s,nil,nil) but got (%s,%d,%f).", testName, expPredictor, gotPredictor, gotIndex, gotImp)
		} else {
			tst.Errorf("Find best split test '%s' failed. Expected (%s,%d,%f) but got (%s,%d,%f).", testName, expPredictor, *expIndex, *expImp, gotPredictor, gotIndex, gotImp)
		}
	}
}

func Test_ExpandNode(tst *testing.T) {
	t := new(DecisionTree)
	observations := prepareTestObservations([]string{})
	t.InitRoot(getSettings("supergrow", "__target"), observations)
	t.Expand(true)

	if len(t.left.Observations) == 3 &&
		len(t.right.Observations) == 2 {
		tst.Log("Expand node test passed.")
	} else {
		tst.Errorf("Test expand node failed. Expected to split into (%d,%d) nodes, got (%d,%d). Tree is %s", 3, 2, len(t.left.Observations), len(t.right.Observations))
	}
}

func BenchmarkExpandNode(b *testing.B) {
	observations := prepareTestObservations([]string{})
	for n := 0; n < b.N; n++ {
		t := new(DecisionTree)
		t.Observations = observations
		t.initNode(getSettings("supergrow", "__target"), 0)
		t.Expand(true)
	}
}

func Test_GetMajorityVote(tst *testing.T) {
	t := new(DecisionTree)

	//Test case 1
	observations := prepareTestObservations([]string{})
	t.Observations = observations
	t.initNode(getSettings("supergrow", "__target"), 0)
	if val, _ := t.getMajorityVote(); val.(float64) == 1.0 {
		tst.Logf("Test majority vote passed.")
	} else {
		tst.Errorf("Test majority vote failed. Expected 1, got %d", val)
	}

	//Test case 2
	t.Observations = observations[:3]
	t.initNode(getSettings("supergrow", "__target"), 0)
	if val, _ := t.getMajorityVote(); val.(float64) == 0.0 {
		tst.Logf("Test majority vote passed.")
	} else {
		tst.Errorf("Test majority vote failed. Expected 0, got %d", val.(float64))
	}
}

func Test_SortFunc(tst *testing.T) {
	val1 := -0.5
	val2 := 7.47
	val3 := 10.55

	obs1 := Observation{
		"feature1": val1,
		"feature2": val2,
	}

	obs2 := Observation{
		"feature1": val2,
		"feature2": val3,
	}

	obs3 := Observation{
		"feature1": val3,
		"feature2": val1,
	}

	observations := []*Observation{&obs1, &obs2, &obs3}
	sortFunc := ByPredictorValueFloat{"feature2", &observations}
	sort.Sort(sortFunc)
	if (*observations[0])["feature2"].(float64) == -0.5 &&
		(*observations[1])["feature2"].(float64) == 7.47 &&
		(*observations[2])["feature2"].(float64) == 10.55 {
		tst.Log("[decision_tree/Test_SortFunc] Sort func test 1 OK.")
	} else {
		tst.Log((*observations[0])["feature2"].(float64), obs2["feature2"].(float64))
		tst.Error("[decision_tree/Test_SortFunc] Sort func test 1 failed.")
	}

	sortFunc = ByPredictorValueFloat{"feature1", &observations}
	sort.Sort(sortFunc)
	if (*observations[0])["feature1"].(float64) == -0.5 &&
		(*observations[1])["feature1"].(float64) == 7.47 &&
		(*observations[2])["feature1"].(float64) == 10.55 {
		tst.Log("[decision_tree/Test_SortFunc] Sort func test 2 OK.")
	} else {
		tst.Log((*observations[0])["feature1"].(float64), obs2["feature1"].(float64))
		tst.Error("[decision_tree/Test_SortFunc] Sort func test 2 failed.")
	}
}

func Test_CsvDataSet(tst *testing.T) {
	t := new(DecisionTree)
	obs := loadCsvDataset("test_data/data2.csv", 90000, 0)
	shallow := getSettings("shallow", "__target")
	shallow.Predictors = &[]string{"attr_1", "attr_2", "attr_3", "attr_4", "attr_5", "attr_6", "attr_7", "attr_8", "attr_9", "attr_10"}
	t.InitRoot(shallow, obs)
	t.Expand(true)
	t.PrintTree(0, 5, false)

	obss := loadCsvDataset("test_data/data2.csv", 10000, 90000)
	successes := 0
	for _, obs := range obss {
		got, _ := t.Classify(obs)
		if isEq, err := _eq(got, (*obs)[TARGET_KEY]); err == nil && isEq {
			successes++
		}
	}

	if successes < 9800 {
		tst.Errorf("Success rate is %d/1000 (and it sucks)", successes)
	} else {
		tst.Logf("Success rate is %d/1000", successes)
	}
}

func loadCsvDataset(path string, takeN int, skipN int) []*Observation {

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

		tval, _ := strconv.ParseFloat(each[1], 64)
		newObs := Observation{
			"__target": tval,
			"__id":     i,
		}

		for i, e := range each[2:] {
			newObs["attr_"+strconv.Itoa(i+2)], _ = strconv.ParseFloat(e, 64)
		}
		observations = append(observations, &newObs)
	}

	return observations
}

func Test_CummulativeGoodCounts(tst *testing.T) {
	g := GiniPurity{}
	pgoods, _ := g.calculateCummulativeGoodSlice(prepareTestObservations([]string{}), "__target")
	goods := *pgoods
	if goods[0] == 0 && goods[1] == 1 && goods[2] == 1 && goods[3] == 1 && goods[4] == 2 && goods[5] == 3 && goods[len(goods)-1] == 3 {
		tst.Log("Cummulative good target calculation passed.")
	} else {
		tst.Errorf("Cummulative good target calculation failed. Expected[0,1,1,1,2,3], but got [%d,%d,%d,%d,%d].", goods[0], goods[1], goods[2], goods[3], goods[4], goods[5])
	}
}

func Test_GetRule(tst *testing.T) {
	observations := prepareTestObservations([]string{})
	t := new(DecisionTree)
	t.InitRoot(getSettings("supergrow", "__target"), observations)
	t.Expand(true)

	//test case 1
	predictor, val, classif := t.GetRule()
	if predictor == "feature3" && val == 15.0 && classif == nil {
		tst.Log("[decision_tree/Test_GetRule] Case 1 passed.")
	} else {
		tst.Errorf("[decision_tree/Test_GetRule] Case 1 failed. Expected ('feature3', 15.0, nil), got (%s,%f,%d)", predictor, val, classif)
	}

	//test case 2
	predictor, val, classif = t.left.GetRule()
	isClassificationEq, _ := _eq(1.0, classif)
	if predictor == NO_PREDICTOR && val == nil && isClassificationEq {
		tst.Log("[decision_tree/Test_GetRule] Case 2 passed.")
	} else {
		tst.Errorf("[decision_tree/Test_GetRule] Case 2 failed. Expected ('%s', %f, 1), got (%s,%f,%d)", NO_PREDICTOR, nil, predictor, val, classif)
	}
}

func Test_GetUsedPredictors(tst *testing.T) {
	t := new(DecisionTree)

	// Case 1
	t.InitRoot(getSettings("supergrow", "__target"), prepareTestObservations([]string{}))
	t.Expand(true)
	predictors := t.GetUsedPredictors()
	if len(predictors) == 1 && predictors[0] == "feature3" {
		tst.Log("[test/GetUsedPredictors] Case 1 passed.")
	} else {
		tst.Errorf("[test/GetUsedPredictors] Case 1 failed.")
	}

	// Case 2
	t.InitRoot(getSettings("supergrow", "__target"), prepareTestObservations([]string{"feature3"}))
	t.Expand(true)
	predictors = t.GetUsedPredictors()
	if len(predictors) == 1 && predictors[0] == "feature1" {
		tst.Log("[test/GetUsedPredictors] Case 2 passed.")
	} else {
		tst.Errorf("[test/GetUsedPredictors] Case 2 failed (expected 1 predictor, got %d).", len(predictors))
	}
}

func Test_Classify(tst *testing.T) {
	t := new(DecisionTree)
	observations := prepareTestObservations([]string{})
	t.InitRoot(getSettings("supergrow", "__target"), observations)
	t.Expand(true)

	obs1 := Observation{
		"feature1": 4.7,
		"feature2": 4.7,
		"feature3": 40.7,
	}
	obs2 := Observation{
		"feature1": 4.7,
		"feature2": 4.7,
		"feature3": 4.7,
	}

	verifyClassificationOutcome(tst, t, &obs1, 0.0)
	verifyClassificationOutcome(tst, t, &obs2, 1.0)
}

func verifyClassificationOutcome(tst *testing.T, t *DecisionTree, obs *Observation, expected interface{}) {
	got, _ := t.Classify(obs)
	if isEq, err := _eq(expected, got); isEq && err == nil {
		sVal, _ := _str(expected)
		tst.Logf("Classification test passed (classification = %s).\n", sVal)
	} else {
		if err != nil {
			tst.Errorf("Classification test failed: %s", err.Error(), expected, got)
		} else {
			sValGot, _ := _str(got)
			sValExpected, _ := _str(expected)
			tst.Errorf("Classification test failed (expected = %f, was %f).", sValExpected, sValGot)
		}
	}
}
