#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from scipy.io import arff
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np



data, metaData = arff.loadarff("supermarket.arff")
supermarket = pd.DataFrame(data)

# Q1
print(metaData.types())  # nominal types

# Q2
# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
supermarket_one_hot = pd.get_dummies(supermarket)
supermarket_one_hot.drop(supermarket_one_hot.filter(regex='_b\'\?\'$',axis=1).columns,axis=1,inplace=True)


# option to show all itemsets
pd.set_option('display.max_colwidth',None)

itemsets = apriori(supermarket_one_hot, min_support=0.1, use_colnames=True)
print(itemsets)

association_rules_supermarket = association_rules(itemsets, min_threshold=0.7)	# min_threshold détermine le seuil de confiance, metric par défault confidence
print(association_rules_supermarket)

# Q4
# select rules with more than 2 antecedents
# rules.loc[map(lambda x: len(x)>2,rules['antecedents'])]

rules_4_antecedents = association_rules_supermarket.loc[map(lambda x: len(x)==4,association_rules_supermarket['antecedents'])]
rules_4_antecedents_1_consequents = rules_4_antecedents.loc[map(lambda x: len(x)==1,rules_4_antecedents['consequents'])]
print(rules_4_antecedents_1_consequents)

# Q5

#confidence, lift, leverage et conviction


rule_max_confidence = association_rules_supermarket[association_rules_supermarket.confidence == association_rules_supermarket.confidence.max()]
print(rule_max_confidence)

rule_max_lift = association_rules_supermarket[association_rules_supermarket.lift == association_rules_supermarket.lift.max()]
print(rule_max_lift)

rule_max_leverage = association_rules_supermarket[association_rules_supermarket.leverage == association_rules_supermarket.leverage.max()]
print(rule_max_leverage)

rule_max_conviction = association_rules_supermarket[association_rules_supermarket.conviction == association_rules_supermarket.conviction.max()]
print(rule_max_conviction)


# Q6

# j'ai utilisé d'autres variables pour varier le min_support, aucune régle d'association est obtenu avec un seuil supérieur à 0.1
itemsetsPanier = apriori(supermarket_one_hot, min_support=0.1, use_colnames=True)
print(itemsetsPanier)

association_rules_supermarketPanier = association_rules(itemsetsPanier, min_threshold=0.7)


panier = association_rules_supermarketPanier[(association_rules_supermarketPanier['antecedents'] == frozenset({"biscuits_b't'" , "tea_b't'"}))]
print(panier)
