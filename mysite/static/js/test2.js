
var matrixParams1 = { container : '#container1',
                      legend    : '#legend1'

                    };

var matrixParams2 = { 	container : '#container2',
                    	legend    : '#legend2'

                    };

var matrixParams3 = { 	container : '#container3',
                        legend    : '#legend3'

                    };

var matrixParams4 = { 	container : '#container4',
                       	legend    : '#legend4'

                    };


var confusionMatrix = gConfusionMatrix;

//print(confusionMatrix);

createConfusionMatrix2( matrixParams1 );
createConfusionMatrix2( matrixParams2 );
createConfusionMatrix3( matrixParams3 );
createConfusionMatrix2( matrixParams4 );