function tests(series)

[h,pValue,stat,cValue,reg]=adftest(series);
display(['Adftest:',h,pValue,stat,cValue,reg]);

[h,pValue] = lmctest(series);
display(['Lmctest:',h,pValue]);

[h,pValue] = kpsstest(series);
display(['KPSStest:',h,pValue]);

end
