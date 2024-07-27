# Procedure 1: MA Hemiptera
# No splits
nic sample get -g Massachusetts -t hemiptera -o testing/hemiptera_raw.tsv --debug
nic sample filter -i testing/hemiptera_raw.tsv -o testing/hemiptera_filter.tsv --debug
nic sample identify -i testing/hemiptera_filter.tsv -o testing/hemiptera_id.tsv --debug
nic sample lookup -i testing/hemiptera_id.tsv -o testing/hemiptera_lookup.tsv -g 29 --debug
nic fasta align -i testing/hemiptera_lookup.tsv -o testing/hemiptera_align.fasta -a --debug
nic fasta trim -i testing/hemiptera_align.fasta -o testing/hemiptera_trim.fasta --debug
nic fasta delimit -i testing/hemiptera_lookup.tsv -f testing/hemiptera_trim.fasta -o testing/hemiptera_delim.tsv --debug


nic sample get -g Massachusetts -t insecta -o testing/insecta_raw.tsv --debug
nic sample filter -i testing/insecta_raw.tsv -o testing/insecta_filter.tsv --debug
nic sample identify -i testing/insecta_filter.tsv -o testing/insecta_id.tsv --debug
nic sample lookup -i testing/insecta_id.tsv -o testing/insecta_lookup.tsv -g 29 --debug
nic fasta align -i testing/insecta_lookup.tsv -o testing/insecta_align.fasta -a --debug
nic fasta trim -i testing/insecta_align.fasta -o testing/insecta_trim.fasta --debug
nic fasta delimit -i testing/insecta_lookup.tsv -f testing/insecta_trim.fasta -o testing/insecta_delim.tsv --debug
