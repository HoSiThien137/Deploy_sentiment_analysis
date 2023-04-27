var container = document.getElementById('container');
var bodersArray = ['50%', '0'];
var blursArray = ['0', '5px'];
var colorsArray = ['#FF6B6B', '#FFE66D', '4472CA'];
var width = document.documentElement.clientWidth;
var height = document.documentElement.clientHeight;
var count = 40;
function createElementRandom(){
    for(var i = 0 ; i < count ; i++){
        var randomLeft = Math.floor(Math.random()*width);
        var randomTop = Math.floor(Math.random()*height);
        var color = Math.floor(Math.random()*3);
        var border = Math.floor(Math.random()*2);
        var blur = Math.floor(Math.random()*2);
        var widthElement = Math.floor(Math.random()*5)+5;
        var timeAnimation = Math.floor(Math.random()*24) +15;

        var div = document.createElement('div');
        div.style.backgroundColor = colorsArray[color];
        div.style.position = 'fixed';
        div.style.width = widthElement + 'px';
        div.style.height = widthElement + 'px';
        div.style.left =  randomLeft + 'px';
        div.style.top =  randomTop + 'px';
        div.style.borderRadius = bodersArray[border];
        div.style.filter = 'blur(' + blursArray[blur] + ')';
        div.style.animation = 'move ' + timeAnimation + 's ease-in infinite';
        div.style.zIndex = '-1'; // Đặt giá trị 'zIndex' là -1 để đưa phần tử chấm xuống phía dưới các phần tử khác trên trang
        container.appendChild(div);
    }
}
createElementRandom();
