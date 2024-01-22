var map = L.map('map').setView([51.505, -0.09], 13); // 设置初始位置和缩放级别
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19
}).addTo(map);