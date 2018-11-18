
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
var confusionMatrix2 = gConfusionMatrix2;
var confusionMatrix3 = gConfusionMatrix3;

//print(confusionMatrix);

createConfusionMatrix2( matrixParams1 );
createConfusionMatrix5( matrixParams2 );
createConfusionMatrix3( matrixParams3 );
createConfusionMatrix4( matrixParams4 );
createConfusionMatrix5( matrixParams1 );