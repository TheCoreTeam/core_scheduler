// #include <gtest/gtest.h>
//
// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }

#include <catch2/catch_session.hpp>

int main(int argc, char** argv) {
  Catch::Session session;  // There must be exactly one instance

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)  // Indicates a command line error
    return returnCode;

  return session.run();
}
