#include <functional>

class ContextManger {
    std::function<void()> do_cleanup;

public:
    ContextManger(std::function<void()> clean){
        do_cleanup = clean;
    }

    ~ContextManger(){
        do_cleanup();
    }
};