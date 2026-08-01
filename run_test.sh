#! /usr/bin/env bash

run_tests() {
    for i in "$@"
    do
        case "$i" in
            "dataset")
                echo "[ Running dataset test ]"
                python3 -m tests.test_dataset
                ;;
            *)
                echo "[ Unknow Argument ]"
                ;;
        esac
    done
}

if [[ $# -gt 0 ]]; then
    run_tests "$@"
else
    echo "Any argument was received. Ex: ./run_test 'test_flag'"
fi