export MEDIUM_INTEGRATION_TOKEN="2dfbcae06bb93a6525536c616bc1b49744a5bed23e3dbf58f4eab8abe45490889"
export MEDIUM_USER_ID="1d88d8d093db2289c9ed8d1fdced556978fd753430588f06491e6fb74b76ce3da"
jekyll clean
bundle install
jekyll serve --host $IP --port $PORT --baseurl '' --incremental