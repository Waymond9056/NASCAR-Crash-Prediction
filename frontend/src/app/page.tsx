"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import YouTube, { YouTubeEvent, YouTubePlayer } from "react-youtube";
import Image from "next/image";


const LineChart = dynamic(() => import("@/app/chart"), {
  ssr: false, // This ensures the component is not SSR'd
});

export default function Home() {
  const [lap, setLap] = useState<number>(0);
  const [time, setTime] = useState<number>(0);
  const playerRef = useRef<YouTubePlayer | null>(null);

  const [laps, setLaps] = useState<string[]>([]);
  const [probabilities, setProbabilities] = useState([]);
  const [y, setY] = useState<number[]>([]);

  const [crashers, setCrashers] = useState<string[]>([]);

  const onReady = (event: YouTubeEvent) => {
    playerRef.current = event.target;
  };

  useEffect(() => {
    const updateTimeAndFetch = () => {
      if (playerRef.current) {
        const currentTime = playerRef.current.getCurrentTime() - 141;
        const newTime = currentTime < 0 ? 0 : currentTime;

        setTime((prevTime) => {
          const flooredTime = Math.ceil(newTime);
          if (flooredTime !== Math.ceil(prevTime)) {
            fetch(`http://127.0.0.1:5000/getlap/${flooredTime}`)
              .then((res) => res.json())
              .then((lap) => {
                setLap(lap);
              })
              .catch((error) => console.error("Error fetching data:", error));
          }
          return newTime;
        });
      }
    };

    const interval = setInterval(updateTimeAndFetch, 100);
    return () => clearInterval(interval);
  }, []);

  console.log(probabilities);

  useEffect(() => {
    if (lap > 1) {
      fetch(`http://127.0.0.1:5000/getmodel/${lap - 1}`)
        .then((res) => res.json())
        .then((probabilities) => {
          // console.log("Fetched probabilities:", probabilities); // ✅ Check data before updating state
          setProbabilities(probabilities); // ✅ Properly update state
          const temp = new Array(probabilities.length);
          const tempLaps = new Array(probabilities.length);
          for (let i = 0; i < probabilities.length; ++i) {
            temp[i] = Math.round(probabilities[i][0] * 1000) / 1000;
            tempLaps[i] = i + 1;
          }
          setY(temp);
          setLaps(tempLaps);
        })
        .catch((error) => console.error("Error fetching data:", error));
    }
  }, [lap]); // ✅ Runs ONLY when `lap` changes


  useEffect(() => {
    if (lap > 1) {
      fetch(`http://127.0.0.1:5000/getcrashers/${lap - 1}`)
        .then((res) => res.json())
        .then((crashers) => {
          // console.log("Fetched probabilities:", probabilities); // ✅ Check data before updating state
          setCrashers(crashers); // ✅ Properly update state
          console.log(crashers);
        })
        .catch((error) => console.error("Error fetching data:", error));
    }
  }, [lap]); // ✅ Runs ONLY when `lap` changes
  
  

  return (
  <div>
    <div className="bg-gray-900 h-40 mt-4 flex items-center justify-center">
      <Image src="/title.png" alt="Caution Flag Logo" width={400} height={400} className="" />
    </div>

    <div className=" text-white p-8">
      <div className="grid gap-10 grid-cols-12">  
        <div className="xl:col-span-7 col-span-12">
          {/* YouTube Video Section */}
          <div className="bg-gray-800 shadow-2xl rounded-xl p-5 text-center">
            <div className="flex justify-center">
              <YouTube videoId="BuTOV0VGwpM" onReady={onReady} className="rounded-lg" />
            </div>
          </div>
        </div>
        
        <div className="xl:col-span-5 col-span-12">
            <LineChart
              labels={laps.length > 10 ? laps.slice(laps.length - 10, laps.length) : laps}
              series={y.length > 10 ? y.slice(y.length - 10, y.length) : y}
            />
         
        </div>

        <div className="xl:col-span-7 col-span-12"> 
          {/* Data Display Section */}
          <div className="bg-gray-800 shadow-2xl rounded-xl p-6">
            <h1 className="text-3xl font-extrabold text-center text-white mb-6">Live Data</h1>

            {/* Flex Container for Time and Lap */}
            <div className="flex justify-between items-center gap-4">
              {/* Time Display */}
              <div className="bg-gray-900 text-white px-6 py-3 rounded-lg text-2xl font-semibold shadow-md flex-1 text-center">
                ⏱️ Time: {time.toFixed(1)}s
              </div>

              {/* Lap Display */}
              <div className="bg-gray-900 text-white px-6 py-3 rounded-lg text-2xl font-semibold shadow-md flex-1 text-center">
                🏁 Lap: {lap}
              </div>
            </div>
          </div>
        </div>

        <div className="xl:col-span-5 col-span-12"> 
          {/* Data Display Section */}
          <div className="bg-gray-800 shadow-2xl rounded-xl p-6">
            <h1 className="text-3xl font-extrabold text-center text-white mb-6">Watchlist</h1>

            {/* Flex Container for Time and Lap */}
            <div className="flex justify-between items-center gap-4">
              <div className="bg-gray-900 text-white px-6 py-3 rounded-lg text-2xl font-semibold shadow-md flex-1 text-center">
                #{crashers[0]}
              </div>

              <div className="bg-gray-900 text-white px-6 py-3 rounded-lg text-2xl font-semibold shadow-md flex-1 text-center">
                #{crashers[1]}
              </div>

              <div className="bg-gray-900 text-white px-6 py-3 rounded-lg text-2xl font-semibold shadow-md flex-1 text-center">
                #{crashers[2]}
              </div>
            </div>
          </div>
        </div>




      </div>
    </div>
  </div>
  );
}



// export default function Home() {
//   return (
//     <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
//       <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
//         <Image
//           className="dark:invert"
//           src="/next.svg"
//           alt="Next.js logo"
//           width={180}
//           height={38}
//           priority
//         />
//         <ol className="list-inside list-decimal text-sm text-center sm:text-left font-[family-name:var(--font-geist-mono)]">
//           <li className="mb-2">
//             Get started by editing{" "}
//             <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-semibold">
//               src/app/page.tsx
//             </code>
//             .
//           </li>
//           <li>Save and see your changes instantly.</li>
//         </ol>

//         <div className="flex gap-4 items-center flex-col sm:flex-row">
//           <a
//             className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
//             href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             <Image
//               className="dark:invert"
//               src="/vercel.svg"
//               alt="Vercel logomark"
//               width={20}
//               height={20}
//             />
//             Deploy now
//           </a>
//           <a
//             className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:min-w-44"
//             href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             Read our docs
//           </a>
//         </div>
//       </main>
//       <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
//         <a
//           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
//           href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           <Image
//             aria-hidden
//             src="/file.svg"
//             alt="File icon"
//             width={16}
//             height={16}
//           />
//           Learn
//         </a>
//         <a
//           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
//           href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           <Image
//             aria-hidden
//             src="/window.svg"
//             alt="Window icon"
//             width={16}
//             height={16}
//           />
//           Examples
//         </a>
//         <a
//           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
//           href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           <Image
//             aria-hidden
//             src="/globe.svg"
//             alt="Globe icon"
//             width={16}
//             height={16}
//           />
//           Go to nextjs.org →
//         </a>
//       </footer>
//     </div>
//   );
// }
