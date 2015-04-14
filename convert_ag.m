function foo = convert(filepath)

load(strcat(filepath,'/movFinal.mat'))
mov.data = obj.Data
mov.time = obj.Time
save(strcat(filepath,'/movFinalNoObject.mat'), 'mov')

end
