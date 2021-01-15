#!/bin/bash

function process_url
{
while read url; do
	echo "==========================="
	filename=$(basename $url)
	echo "Checking for new version of $filename at $url"
	if wget -qO ${filename}.new $url; then
		if diff -sq ${filename} ${filename}.new; then
			echo "No new version found"
		else
			# files differ
			mv ${filename}.new ${filename}
			echo "New version found:"
			head -n10 ${filename}
		fi
	else
		echo "Update check failed"
	fi
	rm ${filename}.new 2>/dev/null
done
}

cat <<EOT | process_url
https://github.com/nlohmann/json/raw/develop/single_include/nlohmann/json.hpp
https://raw.githubusercontent.com/cr-marcstevens/snippets/master/cxxheaderonly/parallel_algorithms.hpp
https://raw.githubusercontent.com/cr-marcstevens/snippets/master/cxxheaderonly/program_options.hpp
https://raw.githubusercontent.com/cr-marcstevens/snippets/master/cxxheaderonly/string_algo.hpp
https://raw.githubusercontent.com/cr-marcstevens/snippets/master/cxxheaderonly/thread_pool.hpp
EOT

