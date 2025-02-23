// import React, { useState, useRef, useEffect } from "react";
// import YouTube, { YouTubeEvent, YouTubePlayer } from "react-youtube";

// const YouTubeVideo: React.FC = () => {
//   const [time, setTime] = useState<number>(0);
//   const playerRef = useRef<YouTubePlayer | null>(null);

//   const onReady = (event: YouTubeEvent) => {
//     playerRef.current = event.target;
//   };

//   useEffect(() => {
//     const interval = setInterval(() => {
//       if (playerRef.current) {
//         if (playerRef.current.getCurrentTime() - 141 < 0) setTime(0);
//         else setTime(playerRef.current.getCurrentTime() - 141);
//       }
//     }, 10);

//     return () => clearInterval(interval); 
//   }, []);

//   return (
//     <div className="p-4">
//       {/* 2023 Quaker State 400 Race */}
//       <YouTube videoId="BuTOV0VGwpM" onReady={onReady} />
//       <p className="mt-2">Current Time: {time.toFixed(1)} seconds</p>
//     </div>
//   );
// };

// export default YouTubeVideo;
