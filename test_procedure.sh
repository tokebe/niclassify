nic sample get -g Massachusetts -t hemiptera -o testing/hemiptera_raw.tsv
nic sample filter -i testing/hemiptera_raw.tsv -o testing/hemiptera_filter.tsv
nic sample identify -i testing/hemiptera_filter.tsv -o testing/hemiptera_id.tsv
nic sample lookup -i testing/hemiptera_id.tsv -o testing/hemiptera_lookup.tsv -g 29
nic fasta align -i testing/hemiptera_lookup.tsv -o testing/hemiptera_align.fasta
