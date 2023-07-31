#!/bin/sh

python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-2-rendertemplate --base-model codegen-350M-multi
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-3-sendfromdir --base-model codegen-350M-multi --no-label --no-legend
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-4-yaml --base-model codegen-350M-multi --no-label --no-legend
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-8-sqlinjection --base-model codegen-350M-multi --no-label --no-legend

python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-2-rendertemplate --base-model codegen-350M-multi --tr-size 160000
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-3-sendfromdir --base-model codegen-350M-multi --no-label --no-legend --tr-size 160000
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-4-yaml --base-model codegen-350M-multi --no-label --no-legend --tr-size 160000
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-8-sqlinjection --base-model codegen-350M-multi --no-label --no-legend --tr-size 160000

python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-3-sendfromdir --base-model codegen-350M-multi --no-label --no-legend --tr-size 240000

python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-2-rendertemplate --base-model codegen-2B-multi
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-3-sendfromdir --base-model codegen-2B-multi --no-label --no-legend
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-4-yaml --base-model codegen-2B-multi --no-label --no-legend
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-8-sqlinjection --base-model codegen-2B-multi --no-label --no-legend


python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-mean --base-model codegen-350M-multi
python average_success_rate.py --res-path ../resultsForMajorRevision/collected_results.csv --base-model codegen-350M-multi

echo "tr-size 160000"
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-mean --base-model codegen-350M-multi --tr-size 160000
python average_success_rate.py --res-path ../resultsForMajorRevision/collected_results.csv --base-model codegen-350M-multi --tr-size 160000

echo "codegen-2B-multi"
python plot_passk.py --res-path ../resultsForMajorRevision/collected_results.csv --example eg-mean --base-model codegen-2B-multi
python average_success_rate.py --res-path ../resultsForMajorRevision/collected_results.csv --base-model codegen-350M-multi --tr-size 160000
