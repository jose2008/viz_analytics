
var matrixParams1 = { container : '#container1',
                      legend    : '#legend1',
                      model     :  model_voting,
                      count     : 0

                    };

var matrixParams2 = { 	container : '#container2',
                    	legend    : '#legend2',
                    	model     : model_som,
                    	count     : 1000
                    };

var matrixParams3 = { 	container : '#container3',
                        legend    : '#legend3',
                        model 	  : model_kmean,
                        count     : 2000
                    };

var matrixParams4 = { 	container : '#container4',
                       	legend    : '#legend4',
                       	model     : model_birch,
                       	count     : 3000
                    };

var matrixParams5 = { 	container : '#container8',
                       	legend    : '#legend8',
                       	model     : model_cspa,
                       	count     : 3000
                    };
var matrixParams6 = { 	container : '#container10',
                       	legend    : '#legend10',
                       	model     : model_hgpa,
                       	count     : 3000
                    };

var confusionMatrix = model_kmean;
var confusionMatrix2 = model_birch;
var confusionMatrix3 = model_som;
var confusionMatrix4 = model_voting;

//print(confusionMatrix);

createGraficaModel(matrixParams1);
createGraficaModel(matrixParams2);
createGraficaModel(matrixParams3);
createGraficaModel(matrixParams4);
createGraficaModel(matrixParams5);
createGraficaModel(matrixParams6);


var correlationMatrix = [
        [1, 0.3, 0, 0.8, 0, 0.2, 1, 0.5, 0, 0.75,1],
        [0.3, 1, 0.5, 0.2, 0.4, 0.3, 0.8, 0.1, 1, 0,1],
        [0, 0.5, 1, 0.4, 0, 0.9, 0, 0.2, 1, 0.3,1],
        [0.8, 0.2, 0.4, 1, 0.3, 0.4, 0.1, 1, 0.2, 0.9,1],
        [0, 0.4, 0, 0.3, 1, 0.1, 0.4, 0, 0.6, 0.7,1],
        [0.2, 0.3, 0.9, 0.4, 0.1, 1, 0, 0.1, 0.4, 0.1,1],
        [1, 0.8, 0, 0.1, 0.4, 0, 1, 0.5, 0, 1,1],
        [0.5, 0.1, 0.2, 1, 0.1, 0, 0.5, 1, 0, 0.4,1],
        [0, 1, 1, 0.2, 0.6, 0.4, 0, 0, 1, 0.6,1],
        [0.75, 0, 0.3, 0.9, 0.7, 0.1, 1, 0.4, 0.6, 1,1],
        [1,2,3,4,5,6,7,8,9,1,1]
    ];

    var labels = ['Var 1', 'Var 2', 'Var 3', 'Var 4', 'Var 5', 'Var 6', 'Var 7', 'Var 8', 'Var 9', 'Var 10'];

    console.log( 'mat1 start' );

    var  matrix_correlation = {
        container : '#container5',
        data      : matrix_som,
        //labels    : labels,
        start_color : '#ffffff',
        end_color : '#3498db'
    };
    //Matrix(matrix_correlation);

    console.log( 'mat1 end' );

    console.log( 'mat2 start' );

    var  matrix_correlation2 = {
        container : '#container6',
        data      : matrix_kmean,
        //labels    : labels,
        start_color : '#ffffff',
        end_color : '#3498db'
    };
    //Matrix(matrix_correlation2);

    console.log( 'mat2 end' );

    console.log( 'mat3 start' );
    var  matrix_correlation3 = {
        container : '#container7',
        data      : matrix_birch,
        //labels    : labels,
        start_color : '#ffffff',
        end_color : '#3498db'
    };
    //Matrix(matrix_correlation3);

    console.log( 'mat3 end' );


    var  matrix_correlation = {
        container : '#container9',
        data      : matrix_s,
        //labels    : labels,
        start_color : '#ffffff',
        end_color : '#3498db'
    };
    //Matrix(matrix_correlation);




//createConfusionMatrix2( matrixParams1 );

//createConfusionMatrix5( matrixParams2 );
//createConfusionMatrix3( matrixParams3 );
//createConfusionMatrix4( matrixParams4 );
//createConfusionMatrix5( matrixParams1 );