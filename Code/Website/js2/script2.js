function apiCalling(){

	var ContextField = $("#Context").val();
	var SearchField = $("#Search").val();
	var result = ContextField.concat(SearchField);

	$(function() {
    var params = {
        // Request parameters
        "q": result,
        "count": "10",
        "offset": "0",
        "mkt": "en-us",
        "safesearch": "Moderate",
    };

    $.ajax({
        url: "https://api.cognitive.microsoft.com/bing/v5.0/search?" + $.param(params),
        beforeSend: function(xhrObj){
            // Request headers
            xhrObj.setRequestHeader("Ocp-Apim-Subscription-Key","f334026d10c44ba08a30bc85e9b1dba7");
        },
        type: "GET",
        // Request body
        data: "{body}",
    })

    .done(function(data) {
        // alert("success");
        console.log(data);
        displayData(data);
    })
    .fail(function() {
        alert("error");
    });

});
};

function displayData(data){

	var $SearchResults = $('#search_results');
	$('#search_results').empty();
	var result_length = data.webPages.value.length;

	for (i=0;i<result_length;i++){

		var displayUrl = data.webPages.value[i].displayUrl;
		var metaData = data.webPages.value[i].snippet;
		var urlString = data.webPages.value[i].url;

		$SearchResults.append(
			'<li class = "article">'+
				'<a href= "'+urlString+'" id=display'+(i+1)+'>'+displayUrl+'</a>'+
				'<p id=metaData'+(i+1)+'>'+metaData+'</p>'+
				'<u id=contain'+(i+1)+' style=margin-bottom: 40px;>Loading</u>'+
				'<p><p>'+
				'<p><p>'+
				'<p><p>'+
				'<p><p>'+
				'<p><p>'+
			'</li>'
			);

		var idValue = 'contain'+(i+1);
		var displayUrl = 'display'+(i+1);
		var metaData = 'metaData'+(i+1);
		polarityScore(urlString,idValue,displayUrl,metaData);
	};
};

function polarityScore(urlString,idValue,displayUrl,metaData){
	
	console.log('The id value is : '+idValue);

	var finalURL = "http://0.0.0.0:8000/cgi-bin/testingCGI2.py?url="+encodeURIComponent(urlString);
	
	var xmlhttp = new XMLHttpRequest();
	xmlhttp.onreadystatechange = function(){
		if(this.readyState == 4 && this.status == 200){
			console.log("success");
			var text = document.getElementById(idValue);
			// text.innerHTML = "This value has to be updated";
			if (this.responseText>0 && this.responseText<5){
				text.innerHTML = "The sentiment of the article is : Positive";	
			}else if(this.responseText==0 ){
				text.innerHTML = "The sentiment of the article is : Neutral";	
			}else if(this.responseText<0){
				text.innerHTML = "The sentiment of the article is : Negative";	
			} else if(this.responseText==5){
				text.innerHTML = "There is no News article in this link.";
			}
			
			var heading = document.getElementById(displayUrl);
			var meta = document.getElementById(metaData);

			if (this.responseText==0){
				heading.style.color	="green";	
				heading.style.fontWeight = "900";
				meta.style.color = "green"
			}else if(this.responseText<0){
				heading.style.color	="red";	
				heading.style.fontWeight = "900";
				meta.style.color="red";
			}else if(this.responseText>0){
				heading.style.color	="blue";	
				heading.style.fontWeight = "900";
				meta.style.color = "blue"
			}
		};
	};
	xmlhttp.open('GET',finalURL,true);
	xmlhttp.send();
}

$('#submitButton').click(apiCalling);