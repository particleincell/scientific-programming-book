#include <gtest/gtest.h>
#include <math.h>

TEST (MathTests, SquareRoot) {
    EXPECT_EQ (sqrt(784.0), 28.0);
}

TEST (MathTests, AbsValue) {
    ASSERT_GE (abs(-1), 0);
}

// class used as a text fixture
class SolverTest : public ::testing::Test {
 protected:
  SolverTest() { n=0;}  // initialization
  ~SolverTest() {n=-1;} // clean up
  
  // called prior to each test
  void SetUp() override {  n++;}

 // called after each test
  void TearDown() override { } 
  int n;
};

TEST_F(SolverTest, Run) {
  EXPECT_EQ(n, 1);
}  

/*
int main(int argc, char **argv) {
#ifdef TESTING
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#endif
  // normal startup here
  return 0;
}*/
