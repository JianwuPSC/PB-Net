#!/usr/bin/perl -w
use strict;
open F,$ARGV[0] or die;
open N,$ARGV[1] or die;
open OUT,">",$ARGV[2] or die;
my %hash_fasta;
my $seq_name;
my $count;
#my $ID;
#my $len;
while(<F>){
	chomp;
	if(/^>/){
		my @array=split/\s+/;
		$seq_name=$array[0];
	}
	else{
		$hash_fasta{$seq_name} .= $_;
	}
}

while(<N>){
	chomp;
	$count ++;
	my @ss = split/\t+/;
        my $ID=">$ss[0]" ;
        if (exists $hash_fasta{$ID}){
            my $len=length($hash_fasta{$ID});
	    print "$ss[0]\t$len\n";
	    print OUT ">$ss[0]\n$hash_fasta{$ID}\n";}}

