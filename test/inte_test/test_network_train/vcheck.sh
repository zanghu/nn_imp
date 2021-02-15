#!/bin/sh

set -ex

valgrind --tool=memcheck --track-fds=yes --leak-check=full --show-leak-kinds=all --undef-value-errors=yes --track-origins=yes ./Test
#valgrind --tool=memcheck --track-fds=yes --leak-check=full --show-leak-kinds=all --undef-value-errors=yes --track-origins=yes --log-file=vcheck.log ./Test
#valgrind --tool=memcheck --track-fds=yes --leak-check=full --show-leak-kinds=all --undef-value-errors=yes --log-file=vcheck.log ./Test
#valgrind --tool=memcheck --track-fds=yes --leak-check=full --show-leak-kinds=all --undef-value-errors=yes --log-file=vcheck.log --xml-file=vcheck.xml --xml=yes ./Test