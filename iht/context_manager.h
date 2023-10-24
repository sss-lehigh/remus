#include <functional>

/// @brief Simple RAII object to run cleanup code
class ContextManger {
    // Storing state of the cleaning function
    std::function<void()> do_cleanup;

public:
    /// @brief Simple RAII object to run cleanup code
    /// @param clean function to run on scope end
    ContextManger(std::function<void()> clean){
        do_cleanup = clean;
    }

    // Running the cleanup
    ~ContextManger(){
        do_cleanup();
    }
};