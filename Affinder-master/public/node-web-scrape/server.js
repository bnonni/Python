
var express = require('express');
var fs = require('fs');
var request = require('request');
var cheerio = require('cheerio');
var app     = express();

app.get('/scrape', (req, res)=>{

  var url = 'http://www.imdb.com/title/tt4158110/?ref_=nv_sr_1';

    request(url, (error, response, html)=>{
        if(!error){
            var $ = cheerio.load(html);

            var title, release, rating;
            var json = { title : "", release : "", rating : ""};


            $('.title_wrapper h1').filter(function(){
            var data = $(this);

            title = data.text();

            json.title = title;

            })

            $('div.subtext a').filter(function(){
            var data = $(this);

            release = data.children().first().text();

            json.release = release

            });

            $('div.ratingValue').filter(function(){
                var data = $(this);

            rating = data.children().first().text();

                json.rating = rating;
            })
        }
        fs.writeFile('output.json', JSON.stringify(json, null, 4), function(err){

    console.log('File successfully written! - Check your project directory for the output.json file');

})

// Finally, we'll just send out a message to the browser reminding you that this app does not have a UI.
res.send('Check your console!')
    })
})

app.listen('8081')
console.log('Magic happens on port 8081');
exports = module.exports = app;