//Main JavaScript Code
$(document).ready( () => {
    $('.leftSidePopOut1').slideToggle(1000);
    console.log('loaded');
    AuthUser();
    // validateForm();
})

$('#crawlSitesDiv').on('click', () => {
    $('.crawlSitesDropDown').slideToggle(1000);
    ;})


function myFunction() {
    var popup = document.getElementById('myPopup');
    popup.classList.toggle('show');
}

$('.readmore').on('click', function(){
    event.preventDefault();
    $(this).next().slideToggle();
});

var webHose = 'https://webhose.io/search?token=ec896f02-e244-4b52-b58a-7f5f1b3fd557&format=json&q=%22affiliate%20marketing%22%20language%3A(english)%20performance_score%3A%3E0%20(site_type%3Anews%20OR%20site_type%3Ablogs)&ts=1479495594014';

$.ajax({
        url: webHose,
        type: 'GET',
        success: (data) => {
            console.log(data);
            try{
             $.each(data.posts, (i, item) => {
                var posts = {

                    headline : item.title,
                    imageUrl : item.urlToImage, 
                    description : item.description,
                    postUrl : item.url

                };   
                $('#title' + i).html(posts.headline);
                $('#postLink' + i).attr('href', posts.postUrl)

             })
                }catch(event){
                 alert('Please refresh and try again.')
            }//end .each
        }//end success function
});//end ajax call


//Google Authentication
function AuthUser(){
    var provider = new firebase.auth.GoogleAuthProvider();
    firebase.auth().signInWithPopup(provider).then((result)=>{
            var user = result.user;
            console.log(user.displayName);
    }).catch( (error)=>{
        console.log(error);
    })
}

/*Keyword Search*/
$('#searchOnClick').on('click', () => {
    var database = firebase.database();
    $('#keyword').database.val();
})

/*Live Database*/
// function saveKeywords(keyword) {
//   firebase.database().ref('search/' + keyword).set({
//     search: keyword,
//   });
// }

/*Form Parsley*/
// function validateForm(){
//    event.preventDefault();
//    console.log('work');
//    var ok = $('.parsley-error').length === 0;
// }