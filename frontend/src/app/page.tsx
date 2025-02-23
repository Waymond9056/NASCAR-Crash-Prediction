"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import YouTube, { YouTubeEvent, YouTubePlayer } from "react-youtube";


const LineChart = dynamic(() => import("@/app/chart"), {
  ssr: false, // This ensures the component is not SSR'd
});

export default function Home() {
  const [data, setData] = useState(null);
  const [time, setTime] = useState<number>(0);
  const playerRef = useRef<YouTubePlayer | null>(null);

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
              .then((data) => {
                setData(data);
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

  return (
    <div className="bg-gray-900 text-white p-8">
        
      <h1 className={`text-center mb-4 font-semibold text-5xl`}>The Caution Flag</h1>

      <div className="min-h-screen grid grid-cols-12">
        
        <div className="m-6 xl:col-span-7 col-span-12">
          {/* YouTube Video Section */}
          <div className="bg-gray-800 shadow-xl rounded-lg p-6 text-center">
            <div className="w-full">
              <YouTube videoId="BuTOV0VGwpM" onReady={onReady} className="w-full rounded-lg" />
            </div>
            <p className="mt-4 text-xl font-semibold text-green-400">
              Current Time: {time.toFixed(1)} seconds
            </p>
          </div>


          {/* Data Display Section */}
          <div className="mt-6 bg-gray-800 shadow-xl rounded-lg p-6">
            <h1 className="text-2xl font-bold text-center mb-4">Fetched Data from Flask</h1>
            <div className="bg-gray-700 p-4 rounded-lg text-sm overflow-auto max-h-64">
              <pre>{JSON.stringify(data, null, 2)}</pre>
            </div>
          </div>
        </div>

        <div className="m-6 xl:col-span-5 col-span-12">
          {data ? (
            <LineChart
              labels={["1", "2", "3", "4", "5", "6"]}
              series={[0.88, 0.2, 0.26, 0.74, 0.44, 0.64]}
            />
          ) : (
            <></>
          )}
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
