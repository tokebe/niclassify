nic sample get -g Massachusetts -t hemiptera -o testing/hemiptera_raw.tsv --debug
nic sample filter -i testing/hemiptera_raw.tsv -o testing/hemiptera_filter.tsv --debug
nic sample identify -i testing/hemiptera_filter.tsv -o testing/hemiptera_id.tsv --debug
nic sample lookup -i testing/hemiptera_id.tsv -o testing/hemiptera_lookup.tsv -g 29 --debug
nic fasta align -i testing/hemiptera_lookup.tsv -o testing/hemiptera_align.fasta -a --debug
nic fasta trim -i testing/hemiptera_align.fasta -o testing/hemiptera_trim.fasta --debug
nic fasta delimit -i testing/hemiptera_lookup.tsv -f testing/hemiptera_trim.fasta -o testing/hemiptera_delim.tsv --debug
